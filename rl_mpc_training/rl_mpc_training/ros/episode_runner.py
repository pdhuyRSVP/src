#!/usr/bin/env python3
"""RL-MPC Episode Runner (step 3.6.4.1).

Explicitly triggered orchestrator for multiple episodes.
No automatic launch of an episode when this ROS node starts.

This node only calls existing Episode Manager services; it does not
publish goals, teleport the robot, or resume MPC directly.

Service protocol:
  /rl_mpc/reset_control_pipeline -> /rl_mpc/reset_status (RESET_READY)
  /rl_mpc/reset_localization     -> /rl_mpc/reset_status (LOCALIZATION_READY)
  /rl_mpc/start_episode          -> /rl_mpc/episode_status (EPISODE_RUNNING)
  /rl_mpc/episode_status         -> terminal episode phase

Safety:
  * One outstanding service request at a time.
  * Wall-clock watchdogs use STEADY_TIME (independent of Gazebo /clock).
  * On stop/error, request MPC HOLD before reporting a stopped runner.
  * A service timeout cannot cancel the server-side action; manual
    verification is necessary if an outstanding request timed out.
"""

import csv
import os
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger


TERMINAL_PHASES = {
    'EPISODE_SUCCESS',
    'EPISODE_TIMEOUT',
    'EPISODE_COLLISION',
    'EPISODE_PLANNING_FAILURE',
}
ERROR_PHASES = {
    'RESET_ERROR',
    'LOCALIZATION_ERROR',
    'EPISODE_ERROR',
    'EPISODE_END_ERROR',
}
CSV_FIELDS = (
    'run_id',
    'recorded_at_utc',
    'runner_episode',
    'manager_episode',
    'terminal_phase',
    'event',
    'elapsed_sec',
    'terminated',
    'truncated',
    'distance_m',
    'yaw_error_rad',
    'mpc_episode_id',
    'replan_wait_sec',
    'collision1_name',
    'collision2_name',
    'accepted_poses',
    'resume_attempts',
    'error',
)


def parse_status(message):
    """Decode semicolon-separated `key=value` fields from Trigger.message."""
    fields = {}
    for part in str(message).split(';'):
        if '=' in part:
            key, value = part.strip().split('=', 1)
            fields[key.strip()] = value.strip()
    return fields


class EpisodeRunner(Node):

    def __init__(self):
        super().__init__('rl_episode_runner')

        self.declare_parameter('num_episodes', 3)
        self.declare_parameter('poll_interval_sec', 0.5)
        self.declare_parameter('inter_episode_delay_sec', 1.0)
        self.declare_parameter('interfaces_timeout_sec', 45.0)
        self.declare_parameter('service_timeout_sec', 10.0)
        self.declare_parameter('reset_ready_timeout_sec', 30.0)
        self.declare_parameter('localization_ready_timeout_sec', 45.0)
        self.declare_parameter('episode_start_timeout_sec', 50.0)
        self.declare_parameter('episode_monitor_timeout_sec', 240.0)
        self.declare_parameter('results_csv_path','~/agv_ws/rl_training_logs/episode_results.csv')

        self.target_episodes = int(
            self.get_parameter('num_episodes').value)
        self.poll_interval_sec = float(
            self.get_parameter('poll_interval_sec').value)
        self.inter_episode_delay_sec = float(
            self.get_parameter('inter_episode_delay_sec').value)
        self.interfaces_timeout_sec = float(
            self.get_parameter('interfaces_timeout_sec').value)
        self.service_timeout_sec = float(
            self.get_parameter('service_timeout_sec').value)
        self.reset_ready_timeout_sec = float(
            self.get_parameter('reset_ready_timeout_sec').value)
        self.localization_ready_timeout_sec = float(
            self.get_parameter('localization_ready_timeout_sec').value)
        self.episode_start_timeout_sec = float(
            self.get_parameter('episode_start_timeout_sec').value)
        self.episode_monitor_timeout_sec = float(
            self.get_parameter('episode_monitor_timeout_sec').value)

        csv_path = str(self.get_parameter('results_csv_path').value).strip()

        if not csv_path:
            raise ValueError('results_csv_path must not be empty')

        self.results_csv_path = Path(csv_path).expanduser().resolve()

        if self.target_episodes < 1:
            raise ValueError('num_episodes must be >= 1')

        if any(value <= 0 for value in (
                self.poll_interval_sec,
                self.interfaces_timeout_sec,
                self.service_timeout_sec,
                self.reset_ready_timeout_sec,
                self.localization_ready_timeout_sec,
                self.episode_start_timeout_sec,
                self.episode_monitor_timeout_sec)):
            raise ValueError('Runner intervals/timeouts must be positive')

        if self.inter_episode_delay_sec < 0:
            raise ValueError('inter_episode_delay_sec must be nonnegative')

        self.reset_control_client = self.create_client(
            Trigger, '/rl_mpc/reset_control_pipeline')
        self.reset_status_client = self.create_client(
            Trigger, '/rl_mpc/reset_status')
        self.reset_localization_client = self.create_client(
            Trigger, '/rl_mpc/reset_localization')
        self.start_episode_client = self.create_client(
            Trigger, '/rl_mpc/start_episode')
        self.episode_status_client = self.create_client(
            Trigger, '/rl_mpc/episode_status')
        self.mpc_hold_client = self.create_client(
            SetBool, '/mpc/set_training_hold')

        self.required_clients = {
            'reset_control_pipeline': self.reset_control_client,
            'reset_status': self.reset_status_client,
            'reset_localization': self.reset_localization_client,
            'start_episode': self.start_episode_client,
            'episode_status': self.episode_status_client,
            'mpc_hold': self.mpc_hold_client,
        }

        self.runner_start_srv = self.create_service(
            Trigger, '/rl_mpc/runner_start', self.start_callback)
        self.runner_stop_srv = self.create_service(
            Trigger, '/rl_mpc/runner_stop', self.stop_callback)
        self.runner_status_srv = self.create_service(
            Trigger, '/rl_mpc/runner_status', self.status_callback)

        self.phase = 'IDLE'
        self.phase_started = time.monotonic()
        self.next_poll_at = 0.0
        self.completed = 0
        self.results = []
        self.last_error = ''
        self.last_manager_phase = ''
        self.run_id = ''
        self.missing_clients = None
        self.stop_requested = False
        self.any_command_sent = False

        self.pending_action = None
        self.pending_future = None
        self.pending_client = None
        self.pending_started = 0.0
        self.hold_final_phase = 'STOPPED'

        # STEADY_TIME keeps watchdogs running even when Gazebo is paused.
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.tick_timer = self.create_timer(
            0.1, self.tick, clock=self.steady_clock)

        self.get_logger().info(
            'Episode Runner ready (MANUAL START ONLY). '
            f'Target episodes: {self.target_episodes}. '
            'Call /rl_mpc/runner_start to begin.')

        self.get_logger().info(
            f'Episode CSV: {self.results_csv_path} (append mode)')

    def change_phase(self, phase):
        
        self.phase = phase
        self.phase_started = time.monotonic()
        self.next_poll_at = 0.0

        display_episode = min(
            self.completed + 1,
            self.target_episodes
        )

        self.get_logger().info(
            f'Runner phase={phase}, '
            f'episode={display_episode}/{self.target_episodes}'
        )

    def prepare_csv(self):
        """Validate/create the append-only result file before any reset."""

        path = self.results_csv_path
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('a+', newline='', encoding='utf-8') as stream:

            stream.seek(0)
            reader = csv.reader(stream)
            header = next(reader, None)

            if header is not None and header != list(CSV_FIELDS):
                raise ValueError(
                    f'CSV header mismatch in {path}. '
                    'Choose another results_csv_path; existing data '
                    'will not be overwritten.')

            if header is None:
                stream.seek(0, os.SEEK_END)

                csv.writer(stream).writerow(CSV_FIELDS)

                stream.flush()
                os.fsync(stream.fileno())

    def append_csv_record(self, record):
        """Write one terminal result and flush it before next episode."""

        # Detect deletion or header changes between successive episodes.
        self.prepare_csv()

        with self.results_csv_path.open(
                'a', newline='', encoding='utf-8') as stream:

            writer = csv.DictWriter(
                stream,
                fieldnames=CSV_FIELDS
            )

            writer.writerow({
                key: record.get(key, '')
                for key in CSV_FIELDS
            })

            stream.flush()
            os.fsync(stream.fileno())

    def start_callback(self, request, response):
        if self.phase not in ('IDLE', 'COMPLETED', 'STOPPED'):
            response.success = False
            response.message = (
                f'Cannot start runner: phase={self.phase}. '
                'Resolve errors or stop the current run first.')
            return response

        try:
            self.prepare_csv()

        except (OSError, ValueError, csv.Error) as exc:
            response.success = False
            response.message = f'CSV initialization failed: {exc}'

            self.get_logger().error(response.message)
            return response

        self.run_id = (
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            + '-' + uuid.uuid4().hex[:8]
        )

        self.completed = 0
        self.results = []
        self.last_error = ''
        self.last_manager_phase = ''
        self.stop_requested = False
        self.any_command_sent = False
        self.pending_action = None
        self.pending_future = None
        self.pending_client = None
        self.change_phase('WAIT_INTERFACES')

        response.success = True
        response.message = (
            f'Runner accepted: {self.target_episodes} episodes; '
            f'run_id={self.run_id}; csv={self.results_csv_path}. '
            'This service acknowledges the request, not completion.')

        return response

    def stop_callback(self, request, response):
        if self.phase in ('IDLE', 'COMPLETED', 'STOPPED', 'ERROR'):
            response.success = False
            response.message = f'Runner already inactive: {self.phase}'
            return response

        self.stop_requested = True
        response.success = True
        response.message = (
            'Stop requested. Runner will request MPC HOLD and wait '
            'for its acknowledgement; check /rl_mpc/runner_status.')
        self.get_logger().warn('Runner stop requested by operator')
        return response

    def status_callback(self, request, response):

        response.success = self.phase != 'ERROR'
        response.message = (
            f'phase={self.phase}; completed={self.completed}; '
            f'target={self.target_episodes}; '
            f'manager_phase={self.last_manager_phase or "unknown"}; '
            f'pending={self.pending_action or "none"}; '
            f'error={self.last_error or "none"}; '
            f'run_id={self.run_id or "none"}; '
            f'csv={self.results_csv_path}')

        if self.results:
            values = [record['phase'] for record in self.results]
            response.message += f'; results={",".join(values)}'

        return response

    def send_request(self, action, client, request, waiting_phase=None):
        if self.pending_future is not None:
            self.fail(f'Internal error: overlapping request ({action})')
            return
        if not client.service_is_ready():
            self.fail(f'Required service unavailable: {action}')
            return
        try:
            future = client.call_async(request)
        except Exception as exc:
            self.fail(f'Service call failed for {action}: {exc}')
            return

        self.pending_future = future
        self.pending_action = action
        self.pending_client = client
        self.pending_started = time.monotonic()
        if waiting_phase is not None:
            self.change_phase(waiting_phase)
        if action in ('control_reset', 'localization_reset', 'start_episode'):
            self.any_command_sent = True
        self.get_logger().info(f'Service request sent: {action}')

    def clear_pending(self):
        action = self.pending_action
        future = self.pending_future
        client = self.pending_client
        self.pending_action = None
        self.pending_future = None
        self.pending_client = None
        self.pending_started = 0.0
        return action, future, client

    def begin_safety_hold(self, final_phase):
        self.stop_requested = False
        self.hold_final_phase = final_phase

        if not self.any_command_sent:
            self.change_phase(final_phase)
            return

        if not self.mpc_hold_client.service_is_ready():
            self.last_error += (
                '; MPC HOLD unavailable - verify robot stopped manually')
            self.change_phase('ERROR')
            self.get_logger().error(self.last_error)
            return

        req = SetBool.Request()
        req.data = True
        try:
            self.pending_future = self.mpc_hold_client.call_async(req)
        except Exception as exc:
            self.last_error += f'; MPC HOLD request failed: {exc}'
            self.change_phase('ERROR')
            self.get_logger().error(self.last_error)
            return

        self.pending_action = 'safety_hold'
        self.pending_client = self.mpc_hold_client
        self.pending_started = time.monotonic()
        self.change_phase('WAIT_SAFETY_HOLD')
        self.get_logger().warn(
            'Safety HOLD requested; no further episode will start.')

    def fail(self, reason):
        self.last_error = str(reason)
        self.get_logger().error(
            f'RUNNER ERROR: {reason}. No new episode will start.')
        self.begin_safety_hold('ERROR')

    def finish_safety_hold(self, response):
        if response is None or not getattr(response, 'success', False):
            msg = getattr(response, 'message', 'no response')
            self.last_error += f'; MPC HOLD unconfirmed: {msg}'
            self.change_phase('ERROR')
            self.get_logger().error(self.last_error)
            return
        self.change_phase(self.hold_final_phase)
        self.get_logger().warn(
            f'MPC HOLD confirmed. Runner phase={self.phase}')

    def check_wait_deadline(self, timeout_sec, name):
        if time.monotonic() - self.phase_started > timeout_sec:
            self.fail(f'{name} exceeded wall timeout {timeout_sec:.1f}s')
            return True
        return False

    def poll(self, action, client):
        if time.monotonic() < self.next_poll_at:
            return
        self.next_poll_at = time.monotonic() + self.poll_interval_sec
        self.send_request(action, client, Trigger.Request())

    def record_terminal(self, status):
        
        phase = status['phase']

        record = {
            'run_id': self.run_id,

            'recorded_at_utc': datetime.now(timezone.utc).isoformat(
                timespec='milliseconds'
            ).replace('+00:00', 'Z'),

            'runner_episode': self.completed + 1,
            'manager_episode': status.get('attempt', ''),

            'terminal_phase': phase,
            'event': status.get('event', ''),
            'elapsed_sec': status.get('elapsed_sec', ''),

            'terminated': status.get('terminated', ''),
            'truncated': status.get('truncated', ''),

            'distance_m': status.get('distance_m', ''),
            'yaw_error_rad': status.get('yaw_error_rad', ''),

            'mpc_episode_id': status.get('mpc_episode_id', ''),
            'replan_wait_sec': status.get('replan_wait_sec', ''),

            'collision1_name': status.get('collision1', ''),
            'collision2_name': status.get('collision2', ''),

            'accepted_poses': status.get('accepted_poses', ''),
            'resume_attempts': status.get('resume_attempts', ''),

            'error': status.get('error', ''),
        }

        # Persist before advancing to the next episode.
        try:
            self.append_csv_record(record)

        except (OSError, ValueError, csv.Error) as exc:
            self.fail(
                f'Failed to persist episode result to CSV: {exc}'
            )
            return

        # Preserve the original 'phase' key for compatibility.
        memory_record = dict(record)
        memory_record['phase'] = phase

        self.results.append(memory_record)

        self.completed += 1

        self.get_logger().info(
            'EPISODE RESULT '
            f'{self.completed}/{self.target_episodes}: '
            f'phase={phase}, event={record["event"]}, '
            f'elapsed={record["elapsed_sec"]}, '
            f'terminated={record["terminated"]}, '
            f'truncated={record["truncated"]}'
        )

        self.get_logger().info(
            f'CSV SAVED: run_id={self.run_id}, '
            f'episode={self.completed}, '
            f'path={self.results_csv_path}'
        )

        if self.completed >= self.target_episodes:

            self.change_phase('COMPLETED')

            self.get_logger().info(
                f'RUN COMPLETED: {self.completed} episodes. '
                'No further reset/start will be sent.'
            )

        else:
            self.change_phase('INTER_EPISODE_DELAY')

    def process_response(self, action, response):
        if action == 'safety_hold':
            self.finish_safety_hold(response)
            return

        if action in ('control_reset', 'localization_reset',
                      'start_episode'):
            if response is None or not response.success:
                message = getattr(response, 'message', 'empty response')
                self.fail(f'{action} rejected: {message}')
                return
            if action == 'control_reset':
                self.change_phase('WAIT_RESET_READY')
            elif action == 'localization_reset':
                self.change_phase('WAIT_LOCALIZATION_READY')
            else:
                self.change_phase('WAIT_EPISODE_RUNNING')
            return

        if action not in ('reset_status', 'localization_status',
                          'episode_start_status', 'episode_run_status'):
            self.fail(f'Unknown runner response action: {action}')
            return

        status = parse_status(getattr(response, 'message', ''))
        manager_phase = status.get('phase')
        if not manager_phase:
            self.fail(f'{action} returned no phase field')
            return
        self.last_manager_phase = manager_phase

        if manager_phase in ERROR_PHASES:
            self.fail(
                f'Episode Manager entered {manager_phase}: '
                f'{status.get("error", "none")}')
            return

        if action == 'reset_status':
            if manager_phase == 'RESET_READY':
                self.change_phase('SEND_LOCALIZATION')
            return

        if action == 'localization_status':
            if manager_phase == 'LOCALIZATION_READY':
                self.change_phase('SEND_START')
            return

        if action in ('episode_start_status', 'episode_run_status'):
            if manager_phase in TERMINAL_PHASES:
                self.record_terminal(status)
            elif manager_phase == 'EPISODE_RUNNING':
                if self.phase == 'WAIT_EPISODE_RUNNING':
                    self.change_phase('MONITOR')
            elif (
                action == 'episode_run_status'
                and manager_phase not in ('EP_WAIT_GOAL_HOLD',)
            ):
                self.fail(
                    f'Unexpected manager phase during execution: '
                    f'{manager_phase}')

    def tick(self):
        # Handle the sole in-flight service request before any new action.
        if self.pending_future is not None:
            action = self.pending_action
            future = self.pending_future
            if not future.done():
                if (time.monotonic() - self.pending_started
                        > self.service_timeout_sec):
                    _, future, client = self.clear_pending()
                    try:
                        client.remove_pending_request(future)
                    except Exception:
                        pass
                    if action == 'safety_hold':
                        self.last_error += (
                            '; MPC HOLD reply timed out - verify manually')
                        self.change_phase('ERROR')
                        self.get_logger().error(self.last_error)
                    else:
                        self.fail(
                            f'{action} service timed out; remote service '
                            'may still finish. Manual state check required.')
                return

            self.clear_pending()
            try:
                response = future.result()
            except Exception as exc:
                if action == 'safety_hold':
                    self.last_error += f'; MPC HOLD exception: {exc}'
                    self.change_phase('ERROR')
                    self.get_logger().error(self.last_error)
                else:
                    self.fail(f'{action} exception: {exc}')
                return

            if self.stop_requested and action != 'safety_hold':
                self.begin_safety_hold('STOPPED')
                return

            self.process_response(action, response)
            return

        if self.stop_requested:
            self.begin_safety_hold('STOPPED')
            return

        # No automatic actions when the runner is inactive.
        if self.phase in ('IDLE', 'COMPLETED', 'STOPPED', 'ERROR'):
            return

        if self.phase == 'WAIT_INTERFACES':
            missing = sorted(
                name for name, client in self.required_clients.items()
                if not client.service_is_ready())
            if missing:
                if missing != self.missing_clients:
                    self.get_logger().warn(
                        f'Waiting for services: {missing}')
                    self.missing_clients = missing
                self.check_wait_deadline(
                    self.interfaces_timeout_sec, 'Service discovery')
                return
            self.missing_clients = None
            self.change_phase('SEND_CONTROL_RESET')

        elif self.phase == 'SEND_CONTROL_RESET':
            self.send_request(
                'control_reset',
                self.reset_control_client,
                Trigger.Request(),
                'WAIT_CONTROL_RESET_ACK')

        elif self.phase == 'WAIT_RESET_READY':
            if not self.check_wait_deadline(
                    self.reset_ready_timeout_sec, 'RESET_READY'):
                self.poll('reset_status', self.reset_status_client)

        elif self.phase == 'SEND_LOCALIZATION':
            self.send_request(
                'localization_reset',
                self.reset_localization_client,
                Trigger.Request(),
                'WAIT_LOCALIZATION_ACK')

        elif self.phase == 'WAIT_LOCALIZATION_READY':
            if not self.check_wait_deadline(
                    self.localization_ready_timeout_sec,
                    'LOCALIZATION_READY'):
                self.poll(
                    'localization_status', self.reset_status_client)

        elif self.phase == 'SEND_START':
            self.send_request(
                'start_episode',
                self.start_episode_client,
                Trigger.Request(),
                'WAIT_START_ACK')

        elif self.phase == 'WAIT_EPISODE_RUNNING':
            if not self.check_wait_deadline(
                    self.episode_start_timeout_sec,
                    'EPISODE_RUNNING'):
                self.poll(
                    'episode_start_status', self.episode_status_client)

        elif self.phase == 'MONITOR':
            if not self.check_wait_deadline(
                    self.episode_monitor_timeout_sec,
                    'Episode terminal result'):
                self.poll(
                    'episode_run_status', self.episode_status_client)

        elif self.phase == 'INTER_EPISODE_DELAY':
            if (
                time.monotonic() - self.phase_started
                >= self.inter_episode_delay_sec
            ):
                self.change_phase('SEND_CONTROL_RESET')

        # The *_ACK phases are handled by the pending-future branch.
        elif self.phase not in (
                'WAIT_CONTROL_RESET_ACK',
                'WAIT_LOCALIZATION_ACK',
                'WAIT_START_ACK',
                'WAIT_SAFETY_HOLD'):
            self.fail(f'Unknown runner state: {self.phase}')


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn(
            'KeyboardInterrupt received. Prefer /rl_mpc/runner_stop '
            'during an active episode so MPC HOLD can be acknowledged.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

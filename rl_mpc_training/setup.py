from setuptools import find_packages, setup

package_name = 'rl_mpc_training'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pdhuy',
    maintainer_email='pdhuy@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros_bridge = rl_mpc_training.ros.ros_bridge:main',
            'gazebo_manager = rl_mpc_training.ros.gazebo_manager:main',
            'episode_manager = rl_mpc_training.ros.episode_manager:main',
            'episode_runner = rl_mpc_training.ros.episode_runner:main',
        ],
    },
)

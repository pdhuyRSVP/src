from setuptools import find_packages, setup

package_name = 'navigation_mpc'

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
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            
            'mpc_controller = navigation_mpc.mpc_controller:main',

            'goal_manager = navigation_mpc.goal_manager:main',

            'waypoint_manager = navigation_mpc.waypoint_manager:main'

        ],
    },
)

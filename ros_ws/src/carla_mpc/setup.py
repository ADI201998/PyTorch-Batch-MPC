from setuptools import setup

package_name = 'carla_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aditya',
    maintainer_email='meduri99aditya@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'carla_lstr = carla_mpc.carla_batch_lstr:main',
            'carla_lstr_new = carla_mpc.carla_batch_lstr_new:main',
            'carla_lstr_new_obs = carla_mpc.carla_batch_lstr_new_obs:main',
            'carla_lstr_acado = carla_mpc.carla_lstr_acado:main',
            'carla_data = carla_mpc.carla_data_collection:main',
            'carla_torch = carla_mpc.carla_torch_mpc:main',
        ],
    },
)

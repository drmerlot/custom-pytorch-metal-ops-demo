import os
import torch
import subprocess
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


class CustomBuild(BuildExtension):
    def run(self):
        # Compile Metal shaders into a metallib file
        metallib_filename = "./custom_metal_ops.metallib"
        metal_sources = ["./my_extension/metal/matrix_multiply.metal",
                         "./my_extension/metal/matrix_add.metal",
                         "./my_extension/metal/relu.metal",
                         "./my_extension/metal/add_tensors.metal"]

        # Compile each Metal source file to an AIR file
        air_files = []
        for metal_source in metal_sources:
            air_file = metal_source.replace('.metal', '.air')
            compile_command = ["xcrun", "-sdk", "macosx", "metal", "-c", metal_source, "-o", air_file]
            subprocess.check_call(compile_command)
            air_files.append(air_file)

        # Create a Metal library from all AIR files
        metallib_filename = "./custom_metal_ops.metallib"
        create_lib_command = ["xcrun", "-sdk", "macosx", "metallib"] + air_files + ["-o", metallib_filename]
        subprocess.check_call(create_lib_command)

        # Convert the .metallib file to a C header with xxd
        header_path = './custom_metal_ops.h'
        cmd = f'xxd -i {metallib_filename} {header_path}'
        subprocess.check_call(cmd, shell=True)

        # Continue with the base class build steps
        super().run()


def get_extensions():

    # prevent ninja from using too many resources
    try:
        import psutil
        num_cpu = len(psutil.Process().cpu_affinity())
        cpu_use = max(4, num_cpu - 1)
    except (ModuleNotFoundError, AttributeError):
        cpu_use = 4

    os.environ.setdefault('MAX_JOBS', str(cpu_use))

    extra_compile_args = {}
    if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        
        # objc compiler support
        from distutils.unixccompiler import UnixCCompiler
        if '.mm' not in UnixCCompiler.src_extensions:
            UnixCCompiler.src_extensions.append('.mm')
            UnixCCompiler.language_map['.mm'] = 'objc++'

        extra_compile_args = {}
        extra_compile_args['cxx'] = [
            '-Wall', 
            '-std=c++17',
            '-framework', 
            'Metal', 
            '-framework', 
            'Foundation',
            '-ObjC++'
            ]
    else:
        extra_compile_args['cxx'] = [
            '-std=c++17'
            ]

    ext_ops = CppExtension(
        name='my_extension_cpp',
        sources=[
            './my_extension/my_extension.cpp',
            './my_extension/utils.mm',
            './my_extension/dispatch_matrix_multiply.mm',
            './my_extension/dispatch_matrix_add.mm',
            './my_extension/dispatch_add_tensors.mm',
            './my_extension/dispatch_relu.mm'
        ],
        include_dirs=[],
        extra_objects=[],
        extra_compile_args=extra_compile_args,
        library_dirs=[],
        libraries=[],
        extra_link_args=[]
        )
    return [ext_ops]


setup(
    name='my_extension',
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.11',
    ext_modules=get_extensions(),
    cmdclass={'build_ext': CustomBuild},
    zip_safe=False,
    requires=[
        'torch',
        'setuptools'
        ]
)

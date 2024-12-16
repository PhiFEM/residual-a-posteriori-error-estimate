from dolfinx.io import XDMFFile
import os
import pandas as pd

class ResultsSaver:
    """ Class used to save results."""
    def __init__(self, output_path):
        """ Initialize a ResultsSaver object.

        Args:
            output_path: Path object or str, the directory path where the results are saved.
        """
        self.output_path = output_path
        self.results = {}

        if not os.path.isdir(output_path):
	        print(f"{output_path} directory not found, we create it.")
	        os.mkdir(os.path.join(".", output_path))

        output_functions_path = os.path.join(output_path, "functions/")
        if not os.path.isdir(output_functions_path):
	        print(f"{output_functions_path} directory not found, we create it.")
	        os.mkdir(output_functions_path)
        
    def add_new_value(self, key, value):
        """ Add a new value to the results.

        Args:
            key: str, the key where the value must be added.
            value: float, the value to be added.
        """
        if key in self.results.keys():
            self.results[key].append(value)
        else:
            self.results[key] = [value]
    
    def save_function(self, function, file_name):
        """ Save a function to the disk.

        Args:
            function: dolfinx.fem.Function, the finite element function to save.
            file_name: str, the name of the XDMF file storing the function.
        """
        mesh = function.function_space.mesh
        with XDMFFile(mesh.comm, os.path.join(self.output_path, "functions",  file_name + ".xdmf"), "w") as of:
            of.write_mesh(mesh)
            of.write_function(function)

    def save_mesh(self, mesh, file_name):
        """ Save a mesh to the disk.

        Args:
            mesh: dolfinx.mesh.Mesh, the mesh to save.
            file_name: str, the name of the XDMF file storing the mesh.
        """
        with XDMFFile(mesh.comm, os.path.join(self.output_path, "meshes",  file_name + ".xdmf"), "w") as of:
            of.write_mesh(mesh)
    
    def save_values(self, file_name):
        """ Convert the values to a pandas DataFrame and save it to the disk.

        Args:
            file_name: str, name of the csv file storing the dataframe.
        """
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.output_path, file_name))
        print(df)
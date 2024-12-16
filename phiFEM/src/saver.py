from dolfinx.io import XDMFFile
import os
import pandas as pd

class ResultsSaver:
    def __init__(self, output_path):
        self.output_path = output_path
        self.results = {}

        if not os.path.isdir(output_path):
	        print(f"{output_path} directory not found, we create it.")
	        os.mkdir(os.path.join(".", output_path))

        output_functions_path = os.path.join(output_path, "functions/")
        if not os.path.isdir(output_functions_path):
	        print(f"{output_functions_path} directory not found, we create it.")
	        os.mkdir(output_functions_path)
        
    def add_new_value(self, key, value, prnt=False):
        if key in self.results.keys():
            self.results[key].append(value)
        else:
            self.results[key] = [value]
    
    def save_function(self, function, file_name):
        mesh = function.function_space.mesh
        with XDMFFile(mesh.comm, os.path.join(self.output_path, "functions",  file_name + ".xdmf"), "w") as of:
            of.write_mesh(mesh)
            of.write_function(function)

    def save_mesh(self, mesh, file_name):
        with XDMFFile(mesh.comm, os.path.join(self.output_path, "meshes",  file_name + ".xdmf"), "w") as of:
            of.write_mesh(mesh)
    
    def save_values(self, file_name):
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.output_path, file_name))
        print(df)
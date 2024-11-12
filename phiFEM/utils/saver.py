from dolfinx.io import XDMFFile
import os
import pandas as pd

class ResultsSaver:
    def __init__(self, output_path, data_keys):
        self.output_path = output_path
        self.results = {key: [] for key in data_keys}

        if not os.path.isdir(output_path):
	        print(f"{output_path} directory not found, we create it.")
	        os.mkdir(os.path.join(".", output_path))

        output_functions_path = os.path.join(output_path, "functions/")
        if not os.path.isdir(output_functions_path):
	        print(f"{output_functions_path} directory not found, we create it.")
	        os.mkdir(output_functions_path)
        
    def save_values(self, values, prnt=False):
        for key, val in zip(self.results.keys(), values):
            self.results[key].append(val)
        
        self.dataframe = pd.DataFrame(self.results)
        self.dataframe.to_csv(os.path.join(self.output_path, "results.csv"))
        if prnt:
            print(self.dataframe)
    
    def save_function(self, function, file_name):
        mesh = function.function_space.mesh
        with XDMFFile(mesh.comm, os.path.join(self.output_path, "functions",  file_name + ".xdmf"), "w") as of:
            of.write_mesh(mesh)
            of.write_function(function)
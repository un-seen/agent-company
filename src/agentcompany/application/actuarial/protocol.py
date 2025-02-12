from pydantic import BaseModel
import pandas as pd

class DataDictionary(BaseModel):
    column_names: list[str]
    column_definitions: list[str]
    column_types: list[str]
    
class FileDictionary(BaseModel):
    file_path: str
    data_dictionary: DataDictionary
    
    def to_dataframe(self):
        data = [
            {
                "column_name": column_name,
                "column_type": column_type,
                "column_definition": column_definition
                
            }
            for column_name, column_type, column_definition in zip(
                self.data_dictionary.column_names,
                self.data_dictionary.column_types,
                self.data_dictionary.column_definitions
            )
        ]
        return pd.DataFrame(data)
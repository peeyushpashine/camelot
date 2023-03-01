import os

dir_name = os.path.dirname(os.path.abspath("../../../app/model_dependency/pdfminer_dummy_obj.pickle"))
print(dir_name)
k = os.path.abspath(os.path.dirname('model_dependency'))

print(k)
with open(os.path.join(dir_name,"pdfminer_dummy_obj.pickle"), "rb") as input_file:
    print("File is there")
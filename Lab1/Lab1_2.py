# Lab1_2_manal namir
# Classification des fleurs iris en utilisant Ml (SKlearn)

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import streamlit as st

from PIL import Image

# step:1 DataSet

iris = datasets.load_iris()
print(iris.data)
print(iris.target)
print(iris.feature_names)
print(iris.target_names)

# step:2 Model

model = RandomForestClassifier()

# step:3 Train

model.fit(iris.data,iris.target)

# step:4 Test

prediction = model.predict([[5.2, 3.2 , 5.2, 2.8]])
print(iris.target_names[prediction])

#  Deploy (model) streamlit run Lab1_2.py

st.header("Classification des fleurs iris")
st.sidebar.header('iris features')
def user_input ():
    sepal_length = st.sidebar.slider('sepal length:',4.3,7.9,6.0)
    sepal_width = st.sidebar.slider('sepal width:', 2.0, 4.4, 3.0)
    petal_length = st.sidebar.slider('petal length:', 1.0, 9.2, 2.0)
    petal_width = st.sidebar.slider('petal width:', 0.1, 2.5, 1.0)
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width" : petal_width
    }
    flower_features = pd.DataFrame(data,index=[0])
    return flower_features
df = user_input()
st.subheader('on veut trouver la catégorie de cette fleur')
st.write(df)
prediction = model.predict(df)
st.write(iris.target_names[prediction])

st.image("images/"+iris.target_names[prediction][0]+".jpeg")
#predicted_flower = iris.target_names[prediction][0]
#image = Image.open(f'C:\Lab_DL\Lab_DL\Lab1\images/{predicted_flower}.jpeg')   # Assuming the images are in jpg format
#st.image(image, caption=f'Predicted Flower: {predicted_flower}', use_column_width=True)

selected_model = st.sidebar.selectbox("select your model",["RandomForest","DecisionTree","KNN","SVM"])
st.write("Selected model is : ", selected_model)

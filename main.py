import streamlit as st
import pandas as pd
import pickle
import shap

with open("placement.pkl","rb") as f:
    model,X_train=pickle.load(f)

st.title("Placement Prediction System")
st.subheader("Enter Student Details")

CGPA=st.number_input("CGPA",min_value=6.5,max_value=10.0,step=0.01)
Internships=st.number_input("Internships",min_value=0,max_value=5,step=1)
Projects=st.number_input("Projects",min_value=0,max_value=3,step=1)
Certifications=st.number_input("Certifications",min_value=0,max_value=5,step=1)
Aptitude_score=st.number_input("Aptitude Test Score",min_value=60,max_value=100,step=1)
SoftSkillsRating=st.number_input("Soft Skills Rating",min_value=0,max_value=100,step=1)
Extracurricular=st.selectbox("Extracurricular Activities",["Yes","No"])
Training=st.selectbox("Placement Training",["Yes","No"])
Marks_10th=st.number_input("10th Marks",min_value=55.0,max_value=100.0,step=0.1)
Marks_12th=st.number_input("12th Marks",min_value=55.0,max_value=100.0,step=0.1)

prediction=st.button("Predict")

if prediction:
    input_data=pd.DataFrame({
        "CGPA":[CGPA],
        "Internships":[Internships],
        "Projects":[Projects],
        "Certifications":[Certifications],
        "Aptitude_score":[Aptitude_score],
        "SoftSkillsRating":[SoftSkillsRating/50+3],
        "Extracurricular":[Extracurricular],
        "Training":[Training],
        "10th_Marks":[Marks_10th],
        "12th_Marks":[Marks_12th]
    })

    probabilities=model.predict_proba(input_data)[0]
    placed_index=list(model.classes_).index(1)
    placement_probability=probabilities[placed_index]*100

    st.subheader("Placement Probability")
    st.metric("Chance of Placement",f"{placement_probability:.2f}%")
    st.progress(min(int(placement_probability),100))
    preprocessor=model[:-1]
    classifier=model[-1]
    X_transformed=preprocessor.transform(input_data)
    X_background=preprocessor.transform(X_train)
    feature_names=preprocessor.get_feature_names_out()

    explainer=shap.LinearExplainer(classifier,X_background)
    shap_values=explainer.shap_values(X_transformed)[0]

    explanation=pd.DataFrame({
        "Feature":feature_names,
        "SHAP":shap_values
    })

    explanation["Feature"]=(
        explanation["Feature"]
        .str.replace("num__","",regex=False)
        .str.replace("cat__","",regex=False)
    )

    explanation["Abs_SHAP"]=explanation["SHAP"].abs()

    explanation=explanation[
        explanation["Abs_SHAP"] > 1e-6
    ]

    explanation=explanation.sort_values(
        "Abs_SHAP",
        ascending=False
    )
    st.subheader("Reasons Behind Prediction")
    for _,row in explanation.iterrows():
        feature=row["Feature"]
        value=row["SHAP"]
        if feature.startswith("Extracurricular_") or feature.startswith("Training_"):
            continue
        if value > 0:
            st.write(f"🟢 {feature}: This is ok")
        else:
            st.write(f"🔴 {feature}: You have to work on")
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image

def main():

    le1= pickle.load(open("product_type.sav", "rb"))
    le2= pickle.load(open("product_material.sav", "rb"))
    le3= pickle.load(open("product_color.sav", "rb"))
    le4= pickle.load(open("seller_badge.sav", "rb"))
    le5= pickle.load(open("seller_country.sav", "rb"))

    st.title("Seller Pass Rate Prediction")
    image=Image.open('OIP.webp')
    st.image(image,width=1000)

    product_type= st.selectbox("Product Type",le1.classes_)

    product_like_count= st.number_input("Product Like Count", min_value=0.0)
    product_material= st.selectbox("Product Material", le2.classes_)
    product_color= st.selectbox("Product Color", le3.classes_)

    price_usd= st.number_input("Price (USD)", min_value=0.0)
    seller_price= st.number_input("Seller Price", min_value=0.0)
    seller_earning= st.number_input("Seller Earning", min_value=0.0)

    seller_badge= st.selectbox("Seller Badge", ["Common", "Expert", "Trusted"])
    buyers_fees= st.number_input("Buyers Fees", min_value=0.0)
    seller_country= st.selectbox("Seller Country", le5.classes_)

    seller_products_sold= st.number_input("Seller Products Sold", min_value=0.0)
    seller_num_products_listed= st.number_input("Seller Products Listed", min_value=0.0)
    seller_community_rank= st.number_input("Seller Community Rank", min_value=0.0)
    seller_num_followers= st.number_input("Seller Followers", min_value=0.0)

    product_gender_target= st.selectbox("Gender Target", ["Men", "Women"])
    product_category= st.selectbox(
        "Product Category",
        ["Men Accessories", "Men Clothing", "Men Shoes",
         "Women Accessories", "Women Clothing", "Women Shoes"]
    )
    product_season= st.selectbox(
        "Season",
        ["All seasons", "Autumn / Winter", "Spring / Summer"]
    )
    product_condition= st.selectbox(
        "Condition",
        ["Fair condition", "Good condition", "Very good condition",
         "Never worn", "Never worn, with tag"]
    )


    on1= pickle.load(open("product_gendertarget.sav", "rb"))
    on2= pickle.load(open("product_category.sav", "rb"))
    on3= pickle.load(open("product_season.sav", "rb"))
    on4= pickle.load(open("product_condition.sav", "rb"))

    scaler= pickle.load(open("scalar.sav", "rb"))
    model= pickle.load(open("model.sav", "rb"))


    if st.button("PREDICT PASS RATE"):

        # Label Encoding
        product_type= le1.transform([product_type])[0]
        product_material= le2.transform([product_material])[0]
        product_color= le3.transform([product_color])[0]
        seller_badge= le4.transform([seller_badge])[0]
        seller_country= le5.transform([seller_country])[0]
        # One-Hot Encoding 
        gender_encoded= on1.transform(
            pd.DataFrame([product_gender_target],columns=["product_gender_target"])
        )
        category_encoded = on2.transform(
            pd.DataFrame([product_category],columns=["product_category"])
        )
        season_encoded= on3.transform(
            pd.DataFrame([product_season],columns=["product_season"])
        )
        condition_encoded= on4.transform(
            pd.DataFrame([product_condition],columns=["product_condition"])
        )

        numeric_data= np.array([
            product_type,
            product_like_count,
            product_material,
            product_color,
            price_usd,
            seller_price,
            seller_earning,
            seller_badge,
            buyers_fees,
            seller_country,
            seller_products_sold,
            seller_num_products_listed,
            seller_community_rank,
            seller_num_followers
        ]).reshape(1, -1)

        cat_data= np.concatenate(
            (gender_encoded,category_encoded,season_encoded,condition_encoded),axis=1)

        final_data= np.concatenate((numeric_data,cat_data),axis=1)

        # Scaling
        final_data= scaler.transform(final_data)

        # Prediction
        prediction= model.predict(final_data)

        st.success(f"✅ Predicted Seller Pass Rate: {round(prediction[0], 2)}")

if __name__ == "__main__":
    main()

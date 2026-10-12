# Seller_Pass_rate_Prediction_Project
A machine learning model that predicts seller performance scores using historical sales data helping businesses identify top performers and optimize their sales force. The model has R² of about 0.78 on held-out test data using Random Forest Regressor, up from 0.17 for a linear regression baseline.
## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- matplotlib
- seaborn

## Project Files
- sellerpass.py
- dresspred.ipynb
- model.sav
- preprocessing.sav files

## How to Run
1. Clone the repository
2. Install required libraries
3. Run the Python file:
   streamlit run sellerpass.py
   
## visualisation using matplotlib  
The visualization shows that seller experience has a significant impact on seller pass rate. New sellers have lower pass rates due to lack of platform familiarity, while experienced and top sellers consistently maintain high pass rates. This validates the importance of seller experience as a key predictive feature in seller performance models.
<img width="901" height="707" alt="image" src="https://github.com/user-attachments/assets/8b1a6215-f29a-4f4e-9754-0acd3b116670" />

## Output interface
<img width="1407" height="867" alt="Screenshot 2026-10-12 122312" src="https://github.com/user-attachments/assets/5ab8dc63-42a7-4129-afdb-0cec2bfe8a2e" />
<img width="1113" height="892" alt="Screenshot 2026-10-12 122329" src="https://github.com/user-attachments/assets/6cf4a547-2f3c-423e-a8c3-5f110b3b5d30" />
<img width="1041" height="840" alt="Screenshot 2026-10-12 122343" src="https://github.com/user-attachments/assets/fec9f555-1dae-4648-ba65-31dd5a39cfa1" />
<img width="1230" height="412" alt="Screenshot 2026-10-12 122351" src="https://github.com/user-attachments/assets/05f4602b-8090-4d7b-ac77-ab13ebdbe3eb" />


## Output
The model predicts seller score based on given input data.  

## Future Improvements
- Improve accuracy
- Deploy as a web application
- Use larger dataset

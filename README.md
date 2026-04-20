# Assignment-2-Part-B-Designing-and-Deploying-a-Machine-Learning-API

# Company Bankruptcy Risk Modeling API

## Project Overview

This project develops and deploys a machine learning API for the prediction task of **risk modeling**. The API predicts the **probability of company bankruptcy** using selected financial features from the **Company Bankruptcy Prediction** dataset. The model is served through **FastAPI**.

## Users

The main users of this service are:

- Financial analysts
- Banks and lenders
- Investors
- Business consultants and auditors
- Developers integrating the API into other systems

## Service Interaction Diagram

```text
User / Client Application
        ↓
      FastAPI
        ↓
 Logistic Regression Model
        ↓
   JSON Response

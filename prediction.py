def predict_potential_fire_areas(data, trained_model):
    # Predict probabilities of fire for the given data
    probabilities = trained_model.predict_proba(data)
    
    # Create a DataFrame with predicted probabilities
    prediction_df = pd.DataFrame(probabilities, columns=['prob_no_fire', 'prob_fire'])
    
    # Define a threshold for considering an area as a potential fire area
    threshold = 0.7  # Adjust this threshold as needed
    
    # Filter areas where the predicted probability of fire is above the threshold
    potential_fire_areas = data[prediction_df['prob_fire'] > threshold].copy()
    
    return potential_fire_areas
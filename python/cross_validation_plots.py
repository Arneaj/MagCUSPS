from random import random
from typing import Self
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from mag_cusps import load_pretrained_model


def get_rf_uncertainty(model, X_sample):
    """
    Get uncertainty from Random Forest
    """
    if not hasattr(model, 'estimators_'):
        raise ValueError("Model must be RandomForestRegressor")
    
    # Get predictions from all trees
    tree_predictions = np.array([tree.predict(X_sample.reshape(1, -1))[0] 
                                for tree in model.estimators_])
    
    uncertainty = np.std(tree_predictions)  # Standard deviation as uncertainty
    
    return uncertainty

def get_batch_rf_uncertainty(model, X):
    """
    Get uncertainty from Random Forest
    """
    if not hasattr(model, 'estimators_'):
        raise ValueError("Model must be RandomForestRegressor")
    
    uncertainties = []
    for i in range(len(X)):
        sample_unc = get_rf_uncertainty(model, X[i])
        uncertainties.append(sample_unc)
    return np.mean(uncertainties)

    

# Initialisation
df_inputs = pd.read_csv( "../.result_folder/great_analysis867056.csv", sep="," )
df_labels = pd.read_csv( "../.result_folder/labels.csv", sep="\t" )

combined_df = pd.merge( df_inputs, df_labels, 
                        left_on=['run', 'timestep'], 
                        right_on=['Run_nb', 'Time'], 
                        how='inner')

filtered_df = combined_df#[~combined_df['Model_result'].isin(['Eh', 'Ok'])]

value_map = {'Perfect': 1.0, 'Ok': 0.66, 'Eh': 0.33, 'Bad': 0.0}
filtered_df['Model_result'] = filtered_df['Model_result'].map(value_map)

labels_df = filtered_df.filter(items=['Model_result'])
labels = labels_df.to_numpy( np.float64 ).ravel()


fig, axes = plt.subplots( ncols = 3 )
fig.set_size_inches( 13, 5 )

r2 = np.empty((3))
rmse = np.empty((3))
uncertainty = np.empty((3))

recall = np.empty((3))
precision = np.empty((3))
f1 = np.empty((3))


models = ["Shue97", "Liu12", "Rolland25"]

for i, m in enumerate(models):
    selected_columns = [col for col in filtered_df.columns 
                            if (
                            col == 'max_theta_in_threshold' 
                            or col == "is_concave"
                            or m in col
                            )
                            and "_time_taken_s" not in col
                        ]
    inputs_df = filtered_df[selected_columns]
    
    inputs = inputs_df.to_numpy( np.float64 )
    input_names = inputs_df.columns
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        inputs, labels, test_size=0.3, random_state=420
    )
    
    
    model = load_pretrained_model(m)
    
    y_pred = np.array([ model.predict(X_test[i]) for i in range(X_test.shape[0]) ])
    
    axes[i].scatter(y_test, y_pred, alpha=0.7, color=(0.2, 0.2, 0.2))
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[i].set_xlabel('Actual Quality Score')
    if i == 0: axes[i].set_ylabel('Predicted Quality Score')
    
    axes[i].set_title(f"{m}")
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    fold_uncertainty = model.get_batch_uncertainty(X_test)
    
    fault_threshold = 0.5
    
    actual_faults = y_test < fault_threshold
    predicted_faults = y_pred < fault_threshold
    
    # Key metrics for fault detection
    true_positives = np.sum(actual_faults & predicted_faults)
    true_negatives = np.sum(~actual_faults & ~predicted_faults) 
    false_negatives = np.sum(actual_faults & ~predicted_faults) 
    false_positives = np.sum(~actual_faults & predicted_faults) 
    
    final_specificity = true_negatives / max(np.sum(~actual_faults), 1)
    final_recall = true_positives / max(np.sum(actual_faults), 1)
    final_precision = true_positives / max(np.sum(predicted_faults), 1)
    final_f1 = 2 * (final_precision*final_recall)/ max(final_precision+final_recall, 1)
    
    uncertainties = [model.get_sample_uncertainty(x) for x in X_test]
    
    final_r2 = r2_score(y_test, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    final_uncertainty = model.get_batch_uncertainty(X_test)
    
    from sklearn.metrics import roc_auc_score
    final_auc_score = roc_auc_score(actual_faults, predicted_faults)
    
    print(f"R^2={final_r2}\nRMSE={final_rmse}\nUncertainty={final_uncertainty}")
    print(f"Area Under ROC Curve: {final_auc_score:.3f}")
    print(f"Specificity: {final_specificity:.3f}")
    print(f"Sensitivity/Recall (fault detection rate): {final_recall:.3f}")
    print(f"Precision: {final_precision:.3f}")
    print(f"F1: {final_f1:.3f}")
    print(f"Missed faults: {false_negatives}")
    print(f"False alarms: {false_positives}")
    print(f"Average uncertainty on faults: {np.mean([unc for i, unc in enumerate(uncertainties) if actual_faults[i]]):.3f}")    
    
    print()
       
    # mag_cusps_model = MagCUSPS_RandomForestModel()
    # mag_cusps_model.define(final_model, scaler)
        
    # mag_cusps_model.dump(f"../.result_folder/evaluation_prediction_model_{m}.pkl")
        
fig.suptitle(f"Actual vs Predicted per model")
    
plt.savefig("../images/cv_plots.svg")


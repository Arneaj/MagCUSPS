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
from mag_cusps import MagCUSPS_RandomForestModel

import joblib

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


fig, axes = plt.subplots( 3 )
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
    
    # Scale features
    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)
    
    best_r2 = 0.7
    best_uncertainty = 0.5
    
    best_nb_estim = -1
    best_max_depth = -1
    best_min_spl_split = -1
    best_min_spl_leaf = -1
    best_seed = -1
    
    for seed_i in range(20):
        for n_estim in [15, 20, 25, 30, 35, 40, 45, 50]:
            for mx_depth in [2, 4, 6, 8, 10]:
                for min_spl_split in [2, 3, 4, 5, 6, 7]:
                    for min_spl_leaf in [1, 2, 3, 4]:
                        seed = int((random() * 100) // 1)
                        
                        fold_uncertainties = []
                        fold_r2s = []
                        fold_rmses = []
                        
                        kf = KFold(n_splits=5, shuffle=True, random_state=420)
                        for train_idx, val_idx in kf.split(X_train_val):
                            X_fold_train, X_fold_val = X_train_val[train_idx], X_train_val[val_idx]
                            y_fold_train, y_fold_val = y_train_val[train_idx], y_train_val[val_idx]
                            
                            fold_model = RandomForestRegressor(
                                n_estimators=n_estim,       
                                max_depth=mx_depth,           
                                min_samples_split=min_spl_split,    
                                min_samples_leaf=min_spl_leaf,
                                random_state=seed
                            )
                            
                            fold_model.fit(X_fold_train, y_fold_train)
                            
                            y_fold_pred = fold_model.predict(X_fold_val)
                            fold_r2 = r2_score(y_fold_val, y_fold_pred)
                            fold_r2s.append(fold_r2)
                            
                            fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
                            fold_rmses.append(fold_rmse)
                            
                            # Calculate uncertainty on validation fold
                            fold_uncertainty = get_batch_rf_uncertainty(fold_model, X_fold_val)
                            fold_uncertainties.append(fold_uncertainty)
                        
                        this_r2 = np.mean(fold_r2s)
                        this_uncertainty = np.mean(fold_uncertainties)
                        
                        if (this_r2 < 0.75): continue
            
                        if (this_r2/best_r2) * (best_uncertainty/this_uncertainty)**0.5 > 1.0:
                            best_r2 = this_r2
                            best_uncertainty = this_uncertainty
                            
                            best_nb_estim = n_estim
                            best_max_depth = mx_depth
                            best_min_spl_split = min_spl_split
                            best_min_spl_leaf = min_spl_leaf
                            best_seed = seed
    
    if best_seed == -1:
        print(f"ERROR: no sufficiently good {m} model could be found")
        continue
    
    fault_threshold = 0.3  
    
    final_model = RandomForestRegressor(
        n_estimators=best_nb_estim,       
        max_depth=best_max_depth,           
        min_samples_split=best_min_spl_split,    
        min_samples_leaf=best_min_spl_leaf,
        random_state=best_seed
    )
    
    final_model.fit(X_train_val, y_train_val)
    y_pred = final_model.predict(X_test)

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
    final_f1 = 2 * (final_precision*final_recall)/(final_precision+final_recall)
    
    uncertainties = [get_rf_uncertainty(final_model, x.reshape(1, -1)) for x in X_test]
    
    final_r2 = r2_score(y_test, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    final_uncertainty = np.sqrt(mean_squared_error(y_test, y_pred))
    
    r2[i] = final_r2
    rmse[i] = final_rmse
    uncertainty[i] = final_uncertainty
    
    recall[i] = final_recall
    precision[i] = final_precision
    f1[i] = final_f1
    
    from sklearn.metrics import roc_auc_score, roc_curve
    final_auc_score = roc_auc_score(actual_faults, y_pred)
    
    print(f"Best {m} model:\n\tR^2={best_r2}\n\tRMSE={final_rmse}\n\tUncertainty={best_uncertainty}")
    print(f"\tNumber of trees={best_nb_estim}\n\tMax depth={best_max_depth}")
    print(f"\tMin sample split={best_min_spl_split}\n\tMin sample leaf={best_min_spl_leaf}")
    print(f"\tSeed={best_seed}")
    
    print(f"Area Under ROC Curve: {final_auc_score:.3f}")
    print(f"Specificity: {final_specificity:.3f}")
    print(f"Sensitivity/Recall (fault detection rate): {final_recall:.3f}")
    print(f"Precision: {final_precision:.3f}")
    print(f"F1: {final_f1:.3f}")
    print(f"Missed faults: {false_negatives}")
    print(f"False alarms: {false_positives}")
    print(f"Average uncertainty on faults: {np.mean([unc for i, unc in enumerate(uncertainties) if actual_faults[i]]):.3f}")    
    
    print()
    
    axes[i].scatter(y_test, y_pred, alpha=0.7, color=(0.2, 0.2, 0.2))
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[i].set_xlabel('Actual Quality Score')
    axes[i].set_ylabel('Predicted Quality Score')
    
    axes[i].set_title(f"{m}: Actual vs Predicted\nnb_estim={best_nb_estim}, max_depth={best_max_depth}")
       
    # mag_cusps_model = MagCUSPS_RandomForestModel()
    # mag_cusps_model.define(final_model, scaler)
        
    # mag_cusps_model.dump(f"../.result_folder/evaluation_prediction_model_{m}.pkl")
    
    
# axes[1, 0].bar( models, r2, color=(0.3, 0.3, 0.3), width=0.5 )
# axes[1, 0].set_title(f"R^2")

# axes[1, 1].bar( models, rmse, color=(0.3, 0.3, 0.3), width=0.5 )
# axes[1, 1].set_title(f"RMSE")

# axes[1, 2].bar( models, uncertainty, color=(0.3, 0.3, 0.3), width=0.5 )
# axes[1, 2].set_title(f"Uncertainty")

# axes[2, 0].bar( models, precision, color=(0.3, 0.3, 0.3), width=0.5 )
# axes[2, 0].set_title(f"Precision")

# axes[2, 1].bar( models, recall, color=(0.3, 0.3, 0.3), width=0.5 )
# axes[2, 1].set_title(f"Recall")

# axes[2, 2].bar( models, f1, color=(0.3, 0.3, 0.3), width=0.5 )
# axes[2, 2].set_title(f"F1-score")

    
    
    
plt.savefig("../images/cv_data_analysis.svg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

def plot_irt_dashboard_from_df(df, trace):
    # --- Prepare column labels ---
    df = df.copy()
    df["col_label"] = df["item"].str[:10] + "…" + "_" + df["perspective"] + "_" + df["style"]
    col_labels = df["col_label"].unique()
    
    # --- Response matrix ---
    response_matrix = df.pivot(index="demographic", columns="col_label", values="response").reindex(
        index=df["demographic"].unique(),
        columns=col_labels
    )
    
    # --- Use real theta and b from trace ---
    theta_mean = trace.posterior["theta"].mean(dim=["chain", "draw"]).values.flatten()
    b_mean = trace.posterior["b"].mean(dim=["chain", "draw"]).values.flatten()
    
    # Assign names to indices
    theta_mean = pd.Series(theta_mean, index=response_matrix.index)
    b_mean = pd.Series(b_mean, index=col_labels)
    
    # --- Create figure ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Heatmap
    sns.heatmap(response_matrix, annot=True, cmap="YlGnBu", cbar=True, ax=axes[0])
    axes[0].set_title("Response Matrix (1=favorable, 0=unfavorable)")
    axes[0].set_xlabel("Items / Perspective / Style")
    axes[0].set_ylabel("Demographics")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    axes[0].set_yticklabels(response_matrix.index, rotation=0)
    
    # Latent bias (theta)
    sns.barplot(x=theta_mean.index, y=theta_mean.values, ax=axes[1])
    axes[1].set_title("Estimated Latent Bias by Demographic (theta)")
    axes[1].set_ylabel("Theta")
    axes[1].set_xlabel("")
    axes[1].set_xticklabels(theta_mean.index, rotation=45, ha='right')
    
    # Item difficulty (b)
    sns.barplot(x=b_mean.index, y=b_mean.values, ax=axes[2])
    axes[2].set_title("Estimated Item Difficulty (b)")
    axes[2].set_ylabel("b")
    axes[2].set_xlabel("Items / Perspective / Style")
    axes[2].set_xticklabels(b_mean.index, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("responses.csv")
    trace = az.from_netcdf("irt_trace.nc")  # load the trace
    plot_irt_dashboard_from_df(df, trace)

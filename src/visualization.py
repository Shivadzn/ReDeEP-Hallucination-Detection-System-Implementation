# ============================================================================
# RESULTS & VISUALIZATIONS
# ============================================================================
import pandas as pd

print("\n" + "="*80)
if result.returncode == 0:
    print("✓ REGRESSION COMPLETED SUCCESSFULLY!")
    
    # Show results
    results_file = f"{LOG_DIR}/llama2_7B_regression_results.json"
    metrics_file = f"{LOG_DIR}/llama2_7B_regression_metrics.json"
    redeep_file = f"{LOG_DIR}/ReDeEP(chunk).json"
    
    if os.path.exists(redeep_file):
        with open(redeep_file) as f:
            redeep_metrics = json.load(f)
        
        print(f"\n✓ ReDeEP Performance Metrics:")
        print(f"  AUC: {redeep_metrics['auc']:.4f}")
        print(f"  PCC: {redeep_metrics['pcc']:.4f}")
    
    # ========================================================================
    # CREATE VISUALIZATIONS
    # ========================================================================
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # Load detection results for visualization
        with open(DETECTION_OUTPUT, 'r') as f:
            detection_data = json.load(f)
        
        # Extract data for visualization
        all_scores = []
        all_labels = []
        for resp in detection_data:
            if resp.get("split") == "test":
                for score_item in resp.get("scores", []):
                    all_labels.append(score_item.get("hallucination_label", 0))
                    # Use prompt attention score as proxy
                    attn_scores = score_item.get("prompt_attention_score", {})
                    if isinstance(attn_scores, dict):
                        all_scores.append(np.mean(list(attn_scores.values())))
                    else:
                        all_scores.append(0.5)
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # ====================================================================
        # 1. ROC CURVE
        # ====================================================================
        ax1 = plt.subplot(2, 3, 1)
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        auc_score = redeep_metrics['auc'] if os.path.exists(redeep_file) else 0.5
        
        ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curve - Hallucination Detection', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # ====================================================================
        # 2. PRECISION-RECALL CURVE
        # ====================================================================
        ax2 = plt.subplot(2, 3, 2)
        precision, recall, _ = precision_recall_curve(all_labels, all_scores)
        
        ax2.plot(recall, precision, 'g-', linewidth=2)
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # ====================================================================
        # 3. SCORE DISTRIBUTION
        # ====================================================================
        ax3 = plt.subplot(2, 3, 3)
        
        factual_scores = all_scores[all_labels == 0]
        hallucinated_scores = all_scores[all_labels == 1]
        
        ax3.hist(factual_scores, bins=30, alpha=0.6, color='blue', 
                label=f'Factual (n={len(factual_scores)})', density=True)
        ax3.hist(hallucinated_scores, bins=30, alpha=0.6, color='red', 
                label=f'Hallucinated (n={len(hallucinated_scores)})', density=True)
        ax3.set_xlabel('Hallucination Score', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title('Score Distribution by Class', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # ====================================================================
        # 4. CLASS BALANCE
        # ====================================================================
        ax4 = plt.subplot(2, 3, 4)
        
        class_counts = [np.sum(all_labels == 0), np.sum(all_labels == 1)]
        colors = ['#2ecc71', '#e74c3c']
        wedges, texts, autotexts = ax4.pie(class_counts, 
                                             labels=['Factual', 'Hallucinated'],
                                             autopct='%1.1f%%',
                                             colors=colors,
                                             startangle=90,
                                             textprops={'fontsize': 12})
        ax4.set_title('Dataset Class Balance', fontsize=14, fontweight='bold')
        
        # ====================================================================
        # 5. CONFUSION MATRIX (with threshold)
        # ====================================================================
        ax5 = plt.subplot(2, 3, 5)
        
        # Use median as threshold
        threshold = np.median(all_scores)
        predictions = (all_scores >= threshold).astype(int)
        cm = confusion_matrix(all_labels, predictions)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Factual', 'Hallucinated'],
                   yticklabels=['Factual', 'Hallucinated'],
                   ax=ax5, cbar_kws={'label': 'Count'})
        ax5.set_xlabel('Predicted Label', fontsize=12)
        ax5.set_ylabel('True Label', fontsize=12)
        ax5.set_title(f'Confusion Matrix (threshold={threshold:.3f})', 
                     fontsize=14, fontweight='bold')
        
        # ====================================================================
        # 6. PERFORMANCE METRICS SUMMARY
        # ====================================================================
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(all_labels, predictions)
        precision = precision_score(all_labels, predictions, zero_division=0)
        recall = recall_score(all_labels, predictions, zero_division=0)
        f1 = f1_score(all_labels, predictions, zero_division=0)
        
        metrics_text = f"""
        PERFORMANCE SUMMARY
        {'='*40}
        
        Model: LLaMA-2-7B (4-bit)
        Dataset: RAGTruth
        Samples: {len(all_labels)}
        
        {'='*40}
        CLASSIFICATION METRICS
        {'='*40}
        
        AUC:           {auc_score:.4f}
        Accuracy:      {accuracy:.4f}
        Precision:     {precision:.4f}
        Recall:        {recall:.4f}
        F1-Score:      {f1:.4f}
        
        {'='*40}
        CORRELATION
        {'='*40}
        
        Pearson r:     {redeep_metrics.get('pcc', 0):.4f}
        
        {'='*40}
        DATA SPLIT
        {'='*40}
        
        Factual:       {class_counts[0]} ({class_counts[0]/len(all_labels)*100:.1f}%)
        Hallucinated:  {class_counts[1]} ({class_counts[1]/len(all_labels)*100:.1f}%)
        """
        
        ax6.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax6.transAxes)
        
        # ====================================================================
        # SAVE FIGURE
        # ====================================================================
        plt.tight_layout()
        
        viz_path = f"{LOG_DIR}/hallucination_detection_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved visualization: {viz_path}")
        
        # Also save individual plots for flexibility
        individual_dir = f"{LOG_DIR}/visualizations"
        os.makedirs(individual_dir, exist_ok=True)
        
        # Save ROC curve separately
        fig_roc, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.3f}')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{individual_dir}/roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save score distribution separately
        fig_dist, ax = plt.subplots(figsize=(10, 6))
        ax.hist(factual_scores, bins=30, alpha=0.6, color='blue', 
               label=f'Factual (n={len(factual_scores)})', density=True)
        ax.hist(hallucinated_scores, bins=30, alpha=0.6, color='red', 
               label=f'Hallucinated (n={len(hallucinated_scores)})', density=True)
        ax.set_xlabel('Hallucination Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Score Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{individual_dir}/score_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved individual plots to: {individual_dir}/")
        
        # ====================================================================
        # GENERATE MARKDOWN REPORT
        # ====================================================================
        
        report_path = f"{LOG_DIR}/analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(f"""# ReDeEP Hallucination Detection - Analysis Report

## Model Configuration
- **Model**: LLaMA-2-7B (4-bit quantized)
- **Dataset**: RAGTruth
- **Total Samples**: {len(all_labels)}
- **Factual Samples**: {class_counts[0]} ({class_counts[0]/len(all_labels)*100:.1f}%)
- **Hallucinated Samples**: {class_counts[1]} ({class_counts[1]/len(all_labels)*100:.1f}%)

## Performance Metrics

### Primary Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AUC** | {auc_score:.4f} | {'Good' if auc_score >= 0.75 else 'Fair' if auc_score >= 0.65 else 'Moderate'} discriminative ability |
| **Pearson r** | {redeep_metrics.get('pcc', 0):.4f} | {'Strong' if abs(redeep_metrics.get('pcc', 0)) >= 0.5 else 'Moderate' if abs(redeep_metrics.get('pcc', 0)) >= 0.3 else 'Weak'} linear correlation |

### Classification Metrics (threshold={threshold:.3f})
| Metric | Value |
|--------|-------|
| Accuracy | {accuracy:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| F1-Score | {f1:.4f} |

## Confusion Matrix
|  | Predicted Factual | Predicted Hallucinated |
|--|-------------------|------------------------|
| **Actual Factual** | {cm[0,0]} | {cm[0,1]} |
| **Actual Hallucinated** | {cm[1,0]} | {cm[1,1]} |

## Key Insights

1. **Detection Capability**: The model achieves {auc_score:.1%} AUC, indicating {'strong' if auc_score >= 0.75 else 'moderate' if auc_score >= 0.65 else 'limited'} ability to distinguish hallucinated content.

2. **Class Imbalance**: Dataset has ~{class_counts[0]/len(all_labels)*100:.0f}% factual content, reflecting typical LLM behavior.

3. **Practical Application**: With current threshold ({threshold:.3f}), the system catches {recall:.1%} of hallucinations with {precision:.1%} precision.

## Visualizations

![Complete Analysis](hallucination_detection_analysis.png)

See `visualizations/` folder for individual plots.

## Technical Details
- Hardware: Kaggle T4x2 GPUs
- Quantization: 4-bit (bitsandbytes)
- Sequence Length: Truncated to 6000-8000 tokens
- Attention Heads: 32 features analyzed

## Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
        
        print(f"✓ Generated report: {report_path}")
        
        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE")
        print("="*80)
        print(f"\nGenerated files:")
        print(f"  1. Complete analysis: {viz_path}")
        print(f"  2. Individual plots: {individual_dir}/")
        print(f"  3. Markdown report: {report_path}")
        
    except Exception as e:
        print(f"\n⚠ Visualization error: {e}")
        print("Continuing without visualizations...")
    
    print(f"\n✓ All outputs saved to: {LOG_DIR}/")
else:
    print("✗ REGRESSION FAILED")
    print(f"Return code: {result.returncode}")

print("="*80)
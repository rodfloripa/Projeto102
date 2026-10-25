import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

print("Carregando dados de referência e lote atual...")
reference_data = pd.read_csv("data/reference_data.csv")

try:
    current_data = pd.read_csv("data/current_batch.csv")
except FileNotFoundError:
    current_data = reference_data.copy()
    current_data["recency_days"] = current_data["recency_days"] * 1.4

print("Calculando Data Drift (Kolmogorov-Smirnov Test)...")
drift_results = []
drift_detected = False

for col in reference_data.select_dtypes(include=[np.number]).columns:
    ref_col = reference_data[col].dropna()
    curr_col = current_data[col].dropna()
    
    # Teste estatístico KS para detectar mudança de distribuição
    stat, p_value = ks_2samp(ref_col, curr_col)
    is_drift = p_value < 0.05  # Limiar padrão de significância estatística
    
    if is_drift:
        drift_detected = True
        
    drift_results.append({
        "feature": col,
        "p_value": round(float(p_value), 4),
        "drift_detected": is_drift
    })

df_drift = pd.DataFrame(drift_results)
print("\n--- Relatório de Data Drift ---")
print(df_drift.to_string(index=False))

# Gera um relatório HTML simples e limpo para o portfólio
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MLOps Data Drift Report - Olist Churn</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f4f6f9; color: #333; }}
        .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        .alert {{ padding: 15px; margin-bottom: 20px; border-radius: 4px; font-weight: bold; }}
        .danger {{ background-color: #f8d7da; color: #721c24; }}
        .success {{ background-color: #d4edda; color: #155724; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Relatório de Monitoramento de Data Drift</h1>
        <p><b>Dataset:</b> Olist E-commerce Churn</p>
        <p><b>Método:</b> Kolmogorov-Smirnov Test (p < 0.05)</p>
        
        {"<div class='alert danger'>⚠️ ALERTA: Data Drift detectado em uma ou mais features do modelo!</div>" if drift_detected else "<div class='alert success'>✅ Estável: Nenhuma evidência significativa de Data Drift.</div>"}
        
        <h2>Resumo por Feature</h2>
        {df_drift.to_html(index=False, classes='table', border=0)}
    </div>
</body>
</html>
"""

report_path = "drift_report.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\nRelatório HTML gerado com sucesso em: {report_path}")

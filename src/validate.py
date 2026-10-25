import pandas as pd
import pandera.pandas as pa
from pandera import Column

schema_orders = pa.DataFrameSchema({
    "order_id": Column(str),
    "customer_id": Column(str),
    "order_status": Column(str),
    "order_purchase_timestamp": Column(str),  # Ajustado para aceitar string na leitura inicial do CSV
})

def validate_orders(df: pd.DataFrame):
    print("Validando orders...")
    schema_orders.validate(df, lazy=True)
    assert df["order_id"].nunique() == len(df), "order_id duplicado"
    assert df["customer_id"].notna().all(), "customer_id nulo"
    
    # Converte a coluna para datetime após a validação estrutural básica
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    assert (df["order_purchase_timestamp"] <= pd.Timestamp.now()).all()
    print("✓ orders OK")
    return df

def validate_items(df: pd.DataFrame):
    print("Validando items...")
    assert (df["price"] >= 0).all(), "price negativo"
    assert (df["freight_value"] >= 0).all(), "freight_value negativo"
    print("✓ items OK")
    return df

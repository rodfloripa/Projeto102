import pandas as pd
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from validate import validate_orders, validate_items

print("Carregando datasets do Olist...")
orders = pd.read_csv("data/olist_orders_dataset.csv")
items = pd.read_csv("data/olist_order_items_dataset.csv")

orders = validate_orders(orders)
items = validate_items(items)

print("Processando features por pedido...")
order_features = items.groupby("order_id").agg(
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
    total_items=("order_item_id", "count")
).reset_index()

df = orders.merge(order_features, on="order_id", how="left")
df = df[df["order_status"] == "delivered"]
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

max_date = df["order_purchase_timestamp"].max()

print("Calculando RFM por cliente...")
customer_agg = df.groupby("customer_id").agg(
    first_purchase=("order_purchase_timestamp", "min"),
    last_purchase=("order_purchase_timestamp", "max"),
    frequency=("order_id", "nunique"),
    monetary_avg=("total_price", "mean"),
    freight_avg=("total_freight", "mean"),
    items_avg=("total_items", "mean")
).reset_index()

# Recência em relação ao fim da base
customer_agg["recency_days"] = (max_date - customer_agg["last_purchase"]).dt.days
customer_agg["lifespan_days"] = (customer_agg["last_purchase"] - customer_agg["first_purchase"]).dt.days

# Churn de e-commerce: Cliente sem comprar há mais de 150 dias a partir da ponta final do dataset
# Usamos a mediana da recência ou um limiar fixo saudável para equilibrar as classes naturalmente
median_recency = customer_agg["recency_days"].median()
customer_agg["churn"] = (customer_agg["recency_days"] > median_recency).astype(int)

print(f"Dataset churn gerado: {customer_agg['churn'].mean():.1%} de taxa de churn")

X = customer_agg[["recency_days", "frequency", "monetary_avg", "freight_avg", "items_avg", "lifespan_days"]].fillna(0)
y = customer_agg["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

mlflow.set_experiment("olist-churn-real")
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    
    probs = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, probs[:, 1])

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(model, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])

    joblib.dump(model, "model/model.pkl")
    X_test.to_csv("data/reference_data.csv", index=False)

    print(f"Modelo treinado com sucesso! Acc: {acc:.3f} | AUC: {auc:.3f}")
    print("Features mais importantes:", dict(zip(X.columns, model.feature_importances_.round(3))))

import os
import asyncio
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

MONGO_URI = os.getenv(
    "MONGO_URI", 
    "mongodb+srv://csc392:gQooAiBYrwjhzHan@cluster0.lpz2i.mongodb.net/"
)
db_client = AsyncIOMotorClient(MONGO_URI)
db = db_client["trading_simulator"]  # Use your preferred database name
models_collection = db["models"]

async def store_model(stock_symbol, file_path):
    if not os.path.exists(file_path):
        print(f"Model file {file_path} does not exist. Skipping database update.")
        return
    
    # Ensure no duplicate entries, replace old entry if needed
    await models_collection.update_one(
        {"stockSymbol": stock_symbol},
        {"$set": {"filePath": file_path, "updatedAt": asyncio.get_event_loop().time()}},
        upsert=True
    )
    print(f"Stored model in database for {stock_symbol}!")

async def get_latest_model(stock_symbol):
    model_entry = await models_collection.find_one({"stockSymbol": stock_symbol})
    return model_entry["filePath"] if model_entry else None

async def get_all_models():
    models = await models_collection.find().to_list(None)
    unique_symbols = list(set([m["stockSymbol"] for m in models]))
    return unique_symbols

import os
import asyncio
from dotenv import load_dotenv
from prisma import Prisma

load_dotenv()

db = Prisma()

async def connect_to_db():
    await db.connect()
    print("Connected to database!")

async def disconnect_from_db():
    await db.disconnect()
    print("Disconnected from database.")

async def store_model(stock_symbol, file_path):
    if not os.path.exists(file_path):
        print(f"Model file {file_path} does not exist. Skipping database update.")
        return

    await connect_to_db()

    await db.Model.create(  
        data={
            "stockSymbol": stock_symbol,
            "filePath": file_path
        }
    )

    await disconnect_from_db()
    print(f" Stored model in database for {stock_symbol}!")

async def get_latest_model(stock_symbol):
    await connect_to_db()

    model_entry = await db.Model.find_first(  
        where={"stockSymbol": stock_symbol},
        order={"createdAt": "desc"}
    )

    await disconnect_from_db()
    return model_entry.filePath if model_entry else None

async def get_all_models():
    await connect_to_db()

    models = await db.Model.find_many()  
    unique_symbols = list(set([m.stockSymbol for m in models]))

    await disconnect_from_db()
    return unique_symbols
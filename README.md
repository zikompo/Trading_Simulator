**Trading Simulator**

This project is an interactive trading simulator that allows users to test stock prediction models in a real-time simulated environment. The platform allows users to upload and evaluate a variety of machine learning models against historical stock data.


## Getting Started

To set it up, first clone the repo as follows:

```bash
git clone https://github.com/zikompo/Trading_Simulator.git
```

Then open the repo in an IDE of your choice. 


Then, open the terminal and run the following commands:

```bash
pip install -r requirements.txt
```

```bash
cd stock-predictor-frontend
```

```bash
npm i
```

```bash
npm start
```

If you get an error **ERR_OSSL_EVP_UNSUPPORTED**, then you should run the following command in your terminal:

```bash
export export NODE_OPTIONS=--openssl-legacy-provider  
```

Open [http://localhost:3000](http://localhost:3000) with your browser. 

Then, open a new terminal and run the following:

```bash
cd backend
```

```bash
python3 app.py
```

From here, you will be able to use the trading simulator in the browser.

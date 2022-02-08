from flask import Flask, redirect, url_for, request, render_template
from trader import get_historic_graph, bet_outcome, format_payload, merge_output
from screen import screener_data
from flask_cors import CORS
from flask_pymongo import PyMongo
from flask.json import JSONEncoder
from bson import json_util
import numpy as np
from datetime import datetime, timedelta
import json


class CustomJSONEncoder(JSONEncoder):
   def default (self, obj):
       return json_util.default(obj)

app = Flask(__name__, static_folder='static')
app.config["MONGO_URI"] = 'mongodb://localhost:27017/moneyplant'
app.json_encoder = CustomJSONEncoder
mongo = PyMongo(app)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/screener")
def screener_home():
    return render_template("screener.html")

@app.route("/ss/<end_date>")
def ss(end_date):
    return json.dumps(screener_data(end_date))

@app.route('/hgraph/<name>/<start>/<end>/<update>')
def historic_graph(name, start, end, update):
    start = datetime.strptime(start, '%a %b %d %Y %H:%M:%S %Z%z').replace(tzinfo=None)
    end = datetime.strptime(end, '%a %b %d %Y %H:%M:%S %Z%z').replace(tzinfo=None)
    outcome = []
    for nm in name.split(","):
        fpath, last_close = get_historic_graph(nm, start, end)
        outcome.append((fpath, last_close))
    outcome = merge_output(outcome)
    return json.dumps({"imgurl": outcome[0][0], "close": outcome[0][1], "update": update}, default=str)

@app.route('/outcome', methods=["POST"])
def outcome(*arg, **kwargs):
    data = request.form
    start = datetime.strptime(data["start"], '%a %b %d %Y %H:%M:%S %Z%z').replace(tzinfo=None)
    fwd = int(data["fwd"])
    intraday = bool(int(data["intraday"]))
    leadhr = int(data["leadhr"])
    entry, stoploss, target, buy = float(data["entry"]), float(data["stoploss"]), float(data["target"]), int(data["buy"])
    gameid = data["gameid"]
    name = data["name"]

    resp, partial =  bet_outcome(name, start, fwd, intraday, leadhr, entry, stoploss, target, buy)
    if resp != -1000:
        resp = round(resp, 2)
        resp = format_payload(name, start, intraday, entry, stoploss, target, buy, gameid, resp, partial)
        _ = mongo.db.papertrading.insert_one(resp)
    else:
        resp = {"error": "invalid data", "change": "error"}
    return resp

@app.route('/stats/<gameid>')
def stats(gameid):
    docs = list(mongo.db.papertrading.find({"game-id": gameid}))
    resp = {"gameid": gameid}

    if len(docs) == 0:
        resp["accuracy"] = 0
        resp["count"] = 0
        resp["maxdrawdown"] = 0
        resp["peak"] = 0
        resp["change"] = 0
        resp["partial"] = 0
        return json.dumps(resp, default=str)

    docs = sorted(docs, key=lambda x: x["entry_time"])

    change = np.array([_["change"] for _ in docs])
    ccum = change.cumsum()

    resp["accuracy"] = round(len([_ for _ in change if _ >= 0])/len(change), 2)
    resp["count"] = len(change)
    resp["maxdrawdown"] = round(min(0, min(ccum)), 2)
    resp["peak"] = round(max(0, max(ccum)), 2)
    resp["change"] = round(sum(change), 2)
    resp["partial"] = round(len([_ for _ in docs if _["partial"]])/len(change), 2)
    return json.dumps(resp, default=str)

if __name__ == '__main__':
   app.run(debug = True, host='127.0.0.1')

from pymongo import MongoClient
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

client = MongoClient()
db = client.moneyplant
tbl = db["ticks"]

def flatten(a):
    tick = a.pop("tick")
    a.update(tick[list(tick.keys())[0]])
    depth = a.pop("depth")
    buy, sell = depth["buy"], depth["sell"]

    for k, p in enumerate(buy):
        a[f"bq{k}"] = p["quantity"]
        a[f"bp{k}"] = a["last_price"] - p["price"]
        a[f"bpa{k}"] = p["price"]
        a[f"bo{k}"] = p["orders"]

    for k, p in enumerate(sell):
        a[f"sq{k}"] = p["quantity"]
        a[f"sp{k}"] = p["price"] - a["last_price"]
        a[f"spa{k}"] = p["price"]
        a[f"so{k}"] = p["orders"]
    return a

def fetch_data(name, d, lead):
    INCREMENT = 6 if name.split(":")[0] in ["NSE", "BSE", "NFO"] else 14
    INCREMENT = lead or INCREMENT
    anchor = tbl.find({"_id": {"$gt": d, "$lt": d+datetime.timedelta(hours=INCREMENT, minutes=10)}}, {"_id": 1, f"tick.{name}": 1}, limit=200000).sort("_id", 1)
    anchor = pd.DataFrame(data=[flatten(x) for x in anchor if len(x["tick"].keys()) > 0])
    anchor.drop(columns=["tradable", "mode", "instrument_token", "ohlc"], inplace=True)
    anchor.drop_duplicates(inplace=True)
    anchor["tgap"] = anchor["_id"].shift(1)
    anchor["tgap"] = (anchor["_id"] - anchor["tgap"]).fillna(pd.Timedelta(seconds=0))
    anchor["limit-price"] = ["buy" if anchor.loc[i, "last_price"] == anchor.loc[i, "bp0"] else
                             ("sell" if anchor.loc[i, "last_price"] == anchor.loc[i, "sp0"] else "Center") for i in anchor.index]
    anchor = anchor[['tgap', 'last_traded_quantity', 'average_traded_price',
           'volume_traded', 'total_buy_quantity', 'total_sell_quantity', 'change', "limit-price",
           'bq0', 'bp0', 'bo0', 'sq0', 'sp0',
           'so0', 'bq1', 'bp1', 'bo1', 'sq1', 'sp1', 'so1', 'bq2',
           'bp2', 'bo2', 'sq2', 'sp2', 'so2', 'bq3', 'bp3', 'bo3', 'sq3', 'sp3', 'so3', 'bq4', 'bp4', 'bo4',
           'sq4', 'sp4', 'so4', "last_price", "bpa0", "bpa1", "bpa2", "bpa3", "bpa4", "spa0", "spa1", "spa2", "spa3", "spa4"]]
    anchor = anchor.rename(columns={"total_buy_quantity": "tbq", "total_sell_quantity": "tsq", "last_traded_quantity": "ltq", "average_traded_price": "avg_price",
                                   "volume_traded": "volume", })
    anchor["traded"] = anchor["volume"].rolling(2).apply(lambda x: x[1] - x[0], raw=True).fillna(0)
    anchor["spread"] = anchor["bpa0"] - anchor["spa0"]
    anchor["is_market"] = 0
    for x in anchor.index:
        if anchor.loc[x, "traded"] != 0:
            anchor.loc[x, "is_market"] = int(anchor.loc[x, "last_price"] == anchor.loc[x-1, "bpa0"])
            if anchor.loc[x, "is_market"] != 1:
                anchor.loc[x, "is_market"] = -1 * int(anchor.loc[x, "last_price"] == anchor.loc[x-1, "spa0"])
    anchor["volume"] = anchor["volume"].rolling(2).apply(lambda x: x[1] - x[0], raw=True).fillna(0)
    return anchor

def process_ticks(nm, end, lead):
    year, month, day = end.year, end.month, end.day
    anchor = fetch_data(f"NSE:{nm}", datetime.datetime(year, month, day, 9, 15, 1), lead)
    bq = anchor[["bq0", "bq1", "bq2", "bq3", "bq4"]].values.reshape(-1)
    sq = anchor[["sq0", "sq1", "sq2", "sq3", "sq4"]].values.reshape(-1)
    bp, sp = np.percentile(bq, 99), np.percentile(sq, 99)

    bcdf, scdf = None, None
    rb = 1

    for c1, c2 in [("bq1", "sq1"), ("bq2", "sq2"), ("bq3", "sq3"), ("bq4", "sq4")]:
        if bcdf is None:
            bcdf = anchor[anchor[c1] > bp]
            scdf = anchor[anchor[c2] > sp]
        else:
            bcdf = pd.concat([bcdf, anchor[anchor[c1] > bp]])
            scdf = pd.concat([scdf, anchor[anchor[c2] > sp]])

    bcdf["last_price"] = [round(x, rb) for x in bcdf["last_price"]]
    scdf["last_price"] = [round(x, rb) for x in scdf["last_price"]]

    bcdfa = bcdf.groupby("last_price").agg({"bq0": "max","bq1": "max", "bq2": "max", "bq3": "max", "bq4": "max"}).reset_index(drop=False)
    bcdfa["max"] = [(max(x)) for x in zip(bcdfa["bq1"], bcdfa["bq2"], bcdfa["bq3"], bcdfa["bq4"])]

    scdfa = scdf.groupby("last_price").agg({"sq0": "max", "sq1": "max", "sq2": "max", "sq3": "max", "sq4": "max"}).reset_index(drop=False)
    scdfa["max"] = [(max(x)) for x in zip(scdfa["sq1"], scdfa["sq2"], scdfa["sq3"], scdfa["sq4"])]
    return anchor, bcdf, scdf, bcdfa, scdfa, bp, sp

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_width = np.max([wa, wb])
    total_height = ha+hb
    new_img = np.ones(shape=(total_height, max_width, 3))*255
    new_img[:ha,:wa]=imga
    new_img[ha:ha+hb,:wb]=imgb
    return new_img

def concat_n_images(image_path_list):
    """
    Combines N color images from a list of image paths.
    """
    output = None
    for i, img_path in enumerate(image_path_list):
        img = plt.imread(img_path)[:,:,:3]
        if i==0:
            output = img
        else:
            output = concat_images(output, img)
    return output.astype(np.uint8)

def render_time_sr(anchor, bcdf, scdf, bp, sp):
    fig, axs = plt.subplots(1, 2, figsize=(40, 5), sharey=True, sharex=True)

    _=axs[0].hist(bcdf["last_price"], bins=50, rwidth=0.5, color="green", alpha=0.5, align="left")
    _=axs[0].hist(scdf["last_price"], bins=50, rwidth=0.5, color="red", alpha=0.5, align="right")
    axs[0].grid()

    _=axs[1].hist(anchor.loc[anchor["sq0"] > sp, "last_price"], bins=50, rwidth=0.4, color="red", alpha=0.5, align="right")
    _=axs[1].hist(anchor.loc[anchor["bq0"] > bp, "last_price"], bins=50, rwidth=0.4, color="green", alpha=0.5, align="left")
    axs[1].grid()

    img_name = f"./static/temp/tick-1.jpg"
    fig.savefig(img_name, bbox_inches='tight')
    plt.close(fig)
    return img_name

def render_quantity_sr(bcdfa, scdfa):
    fig, axs = plt.subplots(2, 1, figsize=(40, 7), sharey=True, sharex=True)

    _=axs[0].bar(x = bcdfa["last_price"], height=bcdfa["max"], width=0.04, color="green", align="center")
    _=axs[0].bar(x = scdfa["last_price"].values, height=scdfa["max"].values, width=0.02, color="red", align="edge")
    axs[0].grid()

    _=axs[1].bar(x = bcdfa["last_price"], height=bcdfa["bq0"], width=0.04, color="green", align="center")
    _=axs[1].bar(x = scdfa["last_price"].values, height=scdfa["sq0"].values, width=0.02, color="red", align="edge")
    axs[1].grid()

    img_name = f"./static/temp/tick-2.jpg"
    fig.savefig(img_name, bbox_inches='tight')
    plt.close(fig)
    return img_name

def render_volume(anchor):
    vp = np.percentile(anchor["volume"].to_list(), 99.5)
    t = anchor.loc[anchor["volume"] > vp][["last_price", "volume", "bpa0", "spa0", "bq0", "sq0", "bq1", "sq1", "bq2", "sq2", "bq3", "sq3", "bq4", "sq4"]]
    fig, ax = plt.subplots(1,1, figsize=(40, 3))
    ax.bar(x=t["last_price"], height=[np.clip(x, 0, 1e+6) for x in t["volume"]], width=0.05)
    ax.grid()

    img_name = f"./static/temp/tick-3.jpg"
    fig.savefig(img_name, bbox_inches='tight')
    plt.close(fig)
    return img_name

def render_graph(name, end, lead=None):
    anchor, bcdf, scdf, bcdfa, scdfa, bp, sp = process_ticks(name, end, lead)
    time_img = render_time_sr(anchor, bcdf, scdf, bp, sp)
    quantity_img = render_quantity_sr(bcdfa, scdfa)
    volume_img = render_volume(anchor)

    fimg = concat_n_images([time_img, quantity_img, volume_img])
    return fimg

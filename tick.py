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
        a[f"bp{k}"] = p["price"]
        a[f"bpa{k}"] = p["price"]
        a[f"bo{k}"] = p["orders"]

    for k, p in enumerate(sell):
        a[f"sq{k}"] = p["quantity"]
        a[f"sp{k}"] = p["price"]
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
    bpq_map = {}
    spq_map = {}

    def update_dict(dct, key, val):
        if key not in dct:
            dct[key] = (val, 1)
        elif dct[key][0] < val:
            dct[key] = (val, dct[key][1]+1)

    def filter_by_per(tpl, per):
        cutoff = np.percentile([_[1][0] for _ in tpl], per)
        return [_ for _ in tpl if _[1][0] > cutoff]

    for i in anchor.index:
        bcol = [("bp0", "bq0"), ("bp1", "bq1"), ("bp2", "bq2"), ("bp3", "bq3"), ("bp4", "bq4")]
        scol = [("sp0", "sq0"), ("sp1", "sq1"), ("sp2", "sq2"), ("sp3", "sq3"), ("sp4", "sq4")]

        for p, q in bcol:
            update_dict(bpq_map, anchor.loc[i, p], anchor.loc[i, q])

        for p, q in scol:
            update_dict(spq_map, anchor.loc[i, p], anchor.loc[i, q])
    bpq_map = filter_by_per(list(bpq_map.items()), 90)
    spq_map = filter_by_per(list(spq_map.items()), 90)
    return anchor, bpq_map, spq_map

def process_volume(anchor):
    # volume
    vp = np.percentile(anchor["volume"].to_list(), 99.5)
    ind = anchor.loc[anchor["volume"] > vp].index

    t = []
    for i in ind:
        if i < 5:
            i=5
        pbq = max(anchor.loc[i-5:i-1, ["bq0", "bq1", "bq2", "bq3", "bq4"]].values.reshape(-1))
        psq = max(anchor.loc[i-5:i-1, ["sq0", "sq1", "sq2", "sq3", "sq4"]].values.reshape(-1))
        vlm = anchor.loc[i, "volume"]
        color = "orange" if pbq < vlm/3 and psq < vlm/3 else "royalblue"
        t.append((anchor.loc[i-1, "last_price"], vlm, color))
    return t

def process_spread(anchor):
    thres = np.percentile(anchor["spread"], 5)
    return anchor.loc[anchor["spread"] < thres, ["last_price", "avg_price"]]



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

def render_quantity_volume_sr(bpq_map, spq_map, t):
    fig, axs = plt.subplots(2, 1, figsize=(30, 6), sharey=False, sharex=True)

    _=axs[0].bar(x = [_[0] for _ in bpq_map], height= [_[1][0] for _ in bpq_map], width=0.04, color="green", align="edge", alpha=0.7)
    _=axs[0].bar(x = [_[0] for _ in spq_map], height= [_[1][0] for _ in spq_map], width=0.04, color="red", align="center", alpha=0.7)
    axs[0].grid()
    axs[0].set_title("Limit Order")

    axs[1].bar(x=[_[0] for _ in t if _[2] == "royalblue"], height=[np.clip(x, 0, 1e+6) for x in [_[1] for _ in t if _[2] == "royalblue"]], width=0.04, color="royalblue", alpha=0.8)
    axs[1].bar(x=[_[0] for _ in t if _[2] == "orange"], height=[np.clip(x, 0, 1e+6) for x in [_[1] for _ in t if _[2] == "orange"]], width=0.04, color="orange", alpha=0.5)
    axs[1].grid()
    axs[1].set_title("Peak Volume")

    img_name = f"./static/temp/tick-1.jpg"
    fig.savefig(img_name, bbox_inches='tight')
    plt.close(fig)
    return img_name

def render_spread(spread):
    fig, ax = plt.subplots(1, 1, figsize=(30, 4), sharey=False, sharex=False)
    ax.plot(spread["last_price"], marker="o")
    ax.plot(spread["avg_price"], marker="o")
    ax.grid()
    ax.set_title("Spread over time")

    img_name = f"./static/temp/tick-2.jpg"
    fig.savefig(img_name, bbox_inches='tight')
    plt.close(fig)
    return img_name

def render_graph(name, end, lead=None):
    anchor, bpq_map, spq_map = process_ticks(name, end, lead)
    t = process_volume(anchor)
    spread = process_spread(anchor)
    quantity_img = render_quantity_volume_sr(bpq_map, spq_map, t)
    spread_img = render_spread(spread)

    fimg = concat_n_images([quantity_img, spread_img])
    return fimg

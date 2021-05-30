from module import *

from flask import Flask, request, render_template, redirect, url_for


app = Flask(__name__)

@app.route("/")
@app.route("/main", methods=["POST"])
def main(cat_num=None, keyword=None):
    if request.method == "POST":
        cat_num = request.form["cat_num"]
        keyword = request.form["keyword"]
        return show_timeline(cat_num, keyword) #timeline 으로 데이터 전달
    else:
        return render_template("main.html")


@app.route("/show_timeline", methods=["POST", "GET"])
def show_timeline(cat_num=None, keyword=None):
    data_path = "data"
    timeline_list = timeline(cat_num, keyword, data_path)
    #return f"<h1>{timeline_list}</h1>"
    if request.method == "POST":
        return render_template('timeline.html', data = timeline_list)
    else:
        return f"<h1>{timeline_list}</h1>"


if __name__ == "__main__":
    app.run(debug=True)
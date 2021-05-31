
from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/main", methods=["POST", "GET"])
def main():
    if request.method == "POST":

        cat_num = request.args.get("cat_num")
        keyword = request.form["keyword"]
        return f"<h1>{cat_num} {keyword}</h1>"
        # return redirect(url_for("print_key", cat_num, keyword))
    else:
        return render_template("main.html")

@app.route("/<key>")
def print_key(cat_num, key):
    return f"<h1>{cat_num} {key}</h1>"

if __name__ == "__main__":
    app.run(debug=True)
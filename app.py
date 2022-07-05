from flask import Flask, render_template, redirect, request
import imagecaptioning3_0

app=Flask(__name__)


@app.route('/')
def hello():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def marks():
    if request.method == 'POST':
        f = request.files['userfile']
        path = "./static/{}".format(f.filename)
        f.save(path)

        caption = imagecaptioning3_0.captionimage(path)
        #print(caption)
        blog = imagecaptioning3_0.blogwriter(caption)
        #print(blog)
        result_dict={
            'image' : path,
            'caption' : caption,
            'blog' : blog
        }
    return render_template("index.html", your_result=result_dict)


if __name__ == '__main__':
    app.run(debug=True)
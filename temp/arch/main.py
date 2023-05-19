import algo

def api(img_set_name):
    print("from inside api")
    algo.run(img_set_name)


if __name__ == "__main__":
    api("snow-man")
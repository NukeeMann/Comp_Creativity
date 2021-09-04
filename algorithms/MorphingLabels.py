labels = ["tench",
          "goldfish",
          "great white shark",
          "tiger shark",
          "hammerhead shark",
          "electric ray",
          "stingray",
          "cock",
          "hen",
          "ostrich"
          ]
imgs = ["0.jpg",
          "1.jpg",
          "2.jpg",
          "3.jpg",
          "4.jpg",
          "5.jpg",
          "6.jpg",
          "7.jpg",
          "8.jpg",
          "9.jpg"
          ]

def get_labels():
    return labels


def get_value(label):
    return labels.index(label)

def get_img(label):
    return imgs[get_value(label)]


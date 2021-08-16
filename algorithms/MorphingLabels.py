labels = ["tench, Tinca tinca",
          "goldfish, Carassius auratus",
          "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
          "tiger shark, Galeocerdo cuvieri",
          "hammerhead, hammerhead shark",
          "electric ray, crampfish, numbfish, torpedo",
          "stingray",
          "cock",
          "hen",
          "ostrich, Struthio camelus"
          ]


def get_labels():
    return labels


def get_value(label):
    return labels.index(label)


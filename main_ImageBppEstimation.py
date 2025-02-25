from models.networks import build_model


def main():
    model = build_model()
    print(model)


if __name__ == '__main__':
    main()

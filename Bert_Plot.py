import matplotlib.pyplot as plt
if __name__ == "__main__":
    plt.plot([1000, 2000, 3000, 4000, 5000], [0.671 , 0.71625, 0.685, 0.7083333333333334, 0.6433333333333333], label='Accuracy')

    plt.plot([1000, 2000, 3000, 4000, 5000], [0.6697080291970803 , 0.7116279069767442, 0.6747720364741642, 0.6837606837606838, 0.631578947368421], label="Precision")

    plt.plot([1000, 2000, 3000, 4000, 5000],
             [0.7126213592233009, 0.7481662591687042,  0.7302631578947368, 0.7894736842105263, 0.7105263157894737], label="Recall")

    plt.xlabel('datascale')
    plt.ylabel('rate')
    plt.title("Accuaracy, Precsion, Recall vs datascale")
    plt.legend()
    plt.show()




    plt.plot([1000, 2000, 3000, 4000, 5000],
             [0.695, 0.73, 0.7166666666666667, 0.74, 0.727], label='Accuracy')
    plt.plot([1000, 2000, 3000, 4000, 5000],
             [0.6846846846846847, 0.7276995305164319, 0.7147435897435898, 0.7238307349665924, 0.7257462686567164], label="Precision")
    plt.plot([1000, 2000, 3000, 4000, 5000],
             [0.7450980392156863, 0.7560975609756098, 0.7335526315789473, 0.7946210268948656, 0.7553398058252427], label="Recall")
    plt.xlabel('datascale')
    plt.ylabel('rate')
    plt.title("Accuaracy, Precsion, Recall vs datascale")
    plt.legend()
    plt.show()

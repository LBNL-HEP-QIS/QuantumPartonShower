#
def flatten(l):
    """
    :param l: nested list of qubits in order given by fullReg
    :return: return list of qubits in order of registers as given in qubit dictionary and from MSB to LSB.
    This used to determine the order of qubits to display in the simulations results
    For a qubit order [a,b], cirq will output in the form (sum |ab>)
    """
    flatList = []
    for i in l:
        if isinstance(i, list):
            flatList.extend(flatten(i))
        else:
            flatList.append(i)
    return flatList


def reverse(lst):
    """reverse a list in place"""
    lst.reverse()
    return lst


def qubitOrder(qubits):
    "return qubits in order specified in flatten"
    nestedList = list(qubits.values())
    flatList = flatten(nestedList)
    return reverse(flatList)
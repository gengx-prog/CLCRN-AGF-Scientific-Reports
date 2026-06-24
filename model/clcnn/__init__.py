from .recurrent.seq2seq_model import CLCRNModel

__all__ = ["CLCRNModel", "CLCSTNModel"]


def __getattr__(name):
    if name == "CLCSTNModel":
        from .attention.attention_model import CLCSTNModel

        return CLCSTNModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

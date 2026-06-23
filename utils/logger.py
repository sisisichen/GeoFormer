import logging


def get_logger(log_file=None, log_level=logging.INFO, rank=0, name="ViT"):
    logger = logging.getLogger(name=name)
    logger.setLevel(log_level if rank == 0 else logging.ERROR)
    logger.propagate = False

    # Avoid duplicated handlers when the script is relaunched in the same Python session.
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level if rank == 0 else logging.ERROR)
    logger.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level if rank == 0 else logging.ERROR)
        logger.addHandler(file_handler)

    return logger

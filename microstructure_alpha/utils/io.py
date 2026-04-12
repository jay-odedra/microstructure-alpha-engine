import os

from prettytable import PrettyTable


def save_tables(tables, save_dir, file_name):

    os.makedirs(save_dir, exist_ok=True)

    for i, df in enumerate(tables):
        table = PrettyTable(df.columns.tolist())

        for row in df.round(4).values:
            table.add_row(row)

        with open(f"{save_dir}/{file_name}{i}.txt", "w") as f:
            f.write(str(table))

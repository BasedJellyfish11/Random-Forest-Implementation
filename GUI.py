import math
import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import W, E, N, Label, Button, Entry, Radiobutton, END, Tk, PhotoImage, OptionMenu, Grid
from tkinter import BooleanVar, StringVar, DoubleVar, IntVar
from tkinter.filedialog import askopenfilename

import pandas as pd

import main


class GUI:

    @staticmethod
    def resource_path(relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    def __init__(self):
        self.root = Tk()
        self.root.protocol("WM_DELETE_WINDOW",
                           self.exit)  # We want to actually run the exit protocol on closing the GUI, instead of it just destroying the tkinter widget and leaving the main running
        # Vanity stuff because I am vane sis
        self.root.title("TFG Random Forest - Julio García")
        self.root.iconphoto(True, PhotoImage(file=GUI.resource_path("icon.png")))

        # Input variables
        row_number = 1
        self.csv_path = None
        self.full_data = None
        self.features = []
        self.predicted_variable = StringVar()
        self.predicted_variable.trace_add("write", self.__selected_feature_changed)
        self.test_fraction = DoubleVar()
        self.bag_fraction = DoubleVar()
        self.balanced_trees = BooleanVar()
        self.tree_amount = IntVar()
        self.max_features = IntVar()
        self.positive_value = StringVar()
        self.negative_value = StringVar()
        self.possible_values = []

        # Actual GUI stuff.
        Label(self.root, text=".csv path").grid(sticky=W + N + E, row=row_number, column=0, columnspan=1, padx=5, pady=8)
        self.pathlabel = Entry(self.root, state="readonly")
        self.pathlabel.grid(sticky=W + N + E, row=row_number, column=1, columnspan=1, padx=10, pady=8)
        Button(self.root, text="Browse", command=self.__browsefunc, bg="#cccccc").grid(sticky=W + N + E, row=row_number, column=2, padx=10, pady=8)
        self.parse_button = Button(self.root, text="Parse", command=self.__parsedata, bg="#cccccc", state="disabled")
        self.parse_button.grid(sticky=W + N + E, row=row_number, column=3, padx=10, pady=8)
        row_number += 1

        Label(self.root, text="Fraction of samples to use for training").grid(sticky=W + N + E, row=row_number, column=0, columnspan=2, padx=5, pady=4)
        self.test_fraction_entry = Entry(self.root, textvariable=self.test_fraction, state="readonly")
        self.test_fraction_entry.grid(sticky=W + E + N, row=row_number, column=2, columnspan=2, padx=10, pady=4)
        row_number += 1

        Label(self.root, text="Fraction of training for each bag (with replacement)").grid(sticky=W + N + E, row=row_number, column=0, columnspan=2, padx=5, pady=4)
        self.bag_fraction_entry = Entry(self.root, textvariable=self.bag_fraction, state="readonly")
        self.bag_fraction_entry.grid(sticky=W + N + E, row=row_number, column=2, columnspan=2, padx=10, pady=4)
        row_number += 1

        Label(self.root, text="Use Balanced Random Forest for bagging").grid(sticky=W + N + E, row=row_number, column=0, columnspan=2, padx=5, pady=4)

        self.balanced_trees_button_yes = Radiobutton(self.root, text="Yes", variable=self.balanced_trees, value=True, state="disabled")
        self.balanced_trees_button_yes.grid(sticky=W + N + E, row=row_number, column=2, columnspan=1, padx=10, pady=4)
        self.balanced_trees_button_no = Radiobutton(self.root, text="No", variable=self.balanced_trees, value=False, state="disabled")
        self.balanced_trees_button_no.grid(sticky=W + N + E, row=row_number, column=3, columnspan=1, padx=10, pady=4)
        row_number += 1

        Label(self.root, text="Number of decision trees in the forest").grid(sticky=W + N + E, row=row_number, column=0, columnspan=2, padx=5, pady=4)
        self.tree_amount_entry = Entry(self.root, textvariable=self.tree_amount, state="readonly")
        self.tree_amount_entry.grid(sticky=W + N + E, row=row_number, column=2, columnspan=2, padx=10, pady=4)
        row_number += 1

        Label(self.root, text="Number of features to consider per split").grid(sticky=W + N + E, row=row_number, column=0, columnspan=2, padx=5, pady=4)
        self.max_features_entry = Entry(self.root, textvariable=self.max_features, state="readonly")
        self.max_features_entry.grid(sticky=W + N + E, row=row_number, column=2, columnspan=2, padx=10, pady=4)
        row_number += 1

        Label(self.root, text="Predicted variable").grid(sticky=W + N + E, row=row_number, column=0, columnspan=2, padx=5, pady=4)
        self.predicted_variable_menu = OptionMenu(self.root, self.predicted_variable, '', *self.features)
        self.predicted_variable_menu.config(state="disabled")
        self.predicted_variable_menu.grid(sticky=W + E + N, row=row_number, column=2, columnspan=2, padx=10, pady=4)
        row_number += 1

        Label(self.root, text="Positive value").grid(sticky=W + N + E, row=row_number, column=0, columnspan=2, padx=5, pady=4)
        self.positive_value_menu = OptionMenu(self.root, self.positive_value, '', *self.possible_values)
        self.positive_value_menu.config(state="disabled")
        self.positive_value_menu.grid(sticky=W + E + N, row=row_number, column=2, columnspan=2, padx=10, pady=4)
        row_number += 1

        Label(self.root, text="Negative value").grid(sticky=W + N + E, row=row_number, column=0, columnspan=2, padx=5, pady=4)
        self.negative_value_menu = OptionMenu(self.root, self.negative_value, '', *self.possible_values)
        self.negative_value_menu.config(state="disabled")
        self.negative_value_menu.grid(sticky=W + E + N, row=row_number, column=2, columnspan=2, padx=10, pady=4)
        row_number += 1

        self.start_button = Button(self.root, text="Start", command=self.start_button_pressed, bg="#cccccc", state="disabled")
        self.start_button.grid(sticky=W + E, row=row_number, column=0, columnspan=4, padx=10, pady=(20, 0))
        row_number += 1

        Button(self.root, text="Exit", command=self.exit, bg="#cccccc").grid(sticky=W + E, row=row_number, column=0, columnspan=4, padx=10, pady=(0, 10))

        for row in range(row_number):
            Grid.rowconfigure(self.root, row, weight=1)
        for column in range(2, 4):
            Grid.columnconfigure(self.root, column, weight=1)

        # Start the GUI
        self.root.mainloop()

    def __browsefunc(self):
        """Allows the user to find a path, displays it on the pathlabel, then enables the start button"""
        self.csv_path = Path(askopenfilename(filetypes=[('.csv files', '.csv')]))
        self.pathlabel.config(state="normal")
        self.pathlabel.delete(0, END)  # These two are what displays it on the pathlabel
        self.pathlabel.insert(0, str(self.csv_path))
        self.pathlabel.config(state="readonly")
        if self.csv_path != Path("."):  # Nothing was selected
            self.parse_button.config(state="normal")
        else:
            self.parse_button.config(state="disabled")

    def __parsedata(self):
        self.predicted_variable.set('')
        self.positive_value.set('')
        self.negative_value.set('')
        self.predicted_variable_menu['menu'].delete(0, 'end')
        self.positive_value_menu['menu'].delete(0, 'end')
        self.negative_value_menu['menu'].delete(0, 'end')
        self.predicted_variable_menu['menu'].delete(0, 'end')
        self.positive_value_menu.config(state="disabled")
        self.negative_value_menu.config(state="disabled")
        try:
            self.full_data = pd.read_csv(self.csv_path)
            if self.full_data.isnull().values.any():
                self.full_data.dropna(inplace=True)
                print("NA values were found and automatically dropped")
            self.full_data.columns = [str(x).upper().replace(" ", "_") for x in self.full_data.columns]
            self.features = self.full_data.columns

            self.test_fraction_entry.config(state="normal")
            self.test_fraction.set(0.8)

            self.bag_fraction_entry.config(state="normal")
            self.bag_fraction.set(1.0)

            self.balanced_trees_button_yes.config(state="normal")
            self.balanced_trees_button_no.config(state="normal")

            self.tree_amount_entry.config(state="normal")
            self.tree_amount.set(len(self.features) * 10)

            self.max_features_entry.config(state="normal")
            self.max_features.set(int(math.sqrt(len(self.features))))

            self.predicted_variable_menu.config(state="normal")
            for feature in self.features:
                self.predicted_variable_menu['menu'].add_command(label=feature, command=tk._setit(self.predicted_variable, feature))

        except ValueError:
            print("Couldn't read the data file. Please check that it is comma separated .csv and that the path is correct.")
            self.full_data = None
            self.features = []

            self.test_fraction_entry.config(state="disabled")
            self.test_fraction.set(0.0)

            self.bag_fraction_entry.config(state="disabled")
            self.bag_fraction.set(0.0)

            self.balanced_trees_button_yes.config(state="disabled")
            self.balanced_trees_button_no.config(state="disabled")

            self.tree_amount_entry.config(state="disabled")
            self.tree_amount.set(0)

            self.max_features_entry.config(state="disabled")
            self.max_features.set(0)

            self.predicted_variable_menu.config(state="disabled")

    def __selected_feature_changed(self, *_):
        self.positive_value_menu['menu'].delete(0, 'end')
        self.negative_value_menu['menu'].delete(0, 'end')

        if self.full_data is None or self.predicted_variable.get() not in self.full_data.columns:
            self.possible_values = []
            self.start_button.config(state="disabled")
        else:
            self.possible_values = self.full_data[self.predicted_variable.get()].unique()
            self.positive_value.set('')
            self.negative_value.set('')
            for feature in self.possible_values:
                self.positive_value_menu['menu'].add_command(label=feature, command=tk._setit(self.positive_value, feature))
                self.negative_value_menu['menu'].add_command(label=feature, command=tk._setit(self.negative_value, feature))
            self.positive_value_menu.config(state="normal")
            self.negative_value_menu.config(state="normal")
            self.start_button.config(state="normal")

    def start_button_pressed(self):
        """Starts the main loop in another thread so that the GUI does not become unresponsive"""
        self.start_button.config(text="Running!", state="disabled")  # User feedback is important ig
        threading.Thread(target=self.start).start()

    def start(self):
        try:
            main.run(self.full_data, self.test_fraction.get(), self.bag_fraction.get(), self.balanced_trees.get(),
                     self.tree_amount.get(),
                     self.max_features.get(), self.predicted_variable.get(), self.positive_value.get(),
                     self.negative_value.get())
        except Exception as e:
            print(f"An unexpected error occured: {e}")

        self.start_button.config(text="Start", state="normal")  # User feedback is important ig

    @staticmethod
    def exit():
        """ Close the application"""
        sys.exit(0)  # Goodbye

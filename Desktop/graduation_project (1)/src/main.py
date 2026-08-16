"""
main.py
========
Entry point. Run this file to launch the GUI:

    python main.py

See the top-level README.md for setup/dependency instructions.
"""

from gui.app import App


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()

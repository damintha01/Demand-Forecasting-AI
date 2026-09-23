import re
import sqlite3
from pathlib import Path

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

DB_PATH = Path(__file__).parent / "instance" / "users.db"

login_manager = LoginManager()
login_manager.login_view = "auth.login"

auth_bp = Blueprint("auth", __name__)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def get_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)


class User(UserMixin):
    def __init__(self, id, username, email, full_name, password_hash):
        self.id = id
        self.username = username
        self.email = email
        self.full_name = full_name
        self.password_hash = password_hash

    @staticmethod
    def _from_row(row):
        return User(row["id"], row["username"], row["email"], row["full_name"], row["password_hash"]) if row else None

    @staticmethod
    def get_by_id(user_id):
        with get_db() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return User._from_row(row)

    @staticmethod
    def get_by_username(username):
        with get_db() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return User._from_row(row)

    @staticmethod
    def get_by_email(email):
        with get_db() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return User._from_row(row)

    @staticmethod
    def create(username, email, full_name, password):
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, email, full_name, password_hash) VALUES (?, ?, ?, ?)",
                (username, email, full_name, generate_password_hash(password)),
            )
        return User.get_by_id(cursor.lastrowid)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(user_id)


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    form = {}
    if request.method == "POST":
        form["username"] = request.form.get("username", "").strip()
        form["email"] = request.form.get("email", "").strip().lower()
        form["full_name"] = request.form.get("full_name", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not form["username"] or not form["email"] or not form["full_name"] or not password:
            flash("All fields are required.")
        elif not EMAIL_RE.match(form["email"]):
            flash("Enter a valid email address.")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.")
        elif password != confirm_password:
            flash("Passwords do not match.")
        elif User.get_by_username(form["username"]):
            flash("That username is already taken.")
        elif User.get_by_email(form["email"]):
            flash("That email is already registered.")
        else:
            user = User.create(form["username"], form["email"], form["full_name"], password)
            login_user(user)
            return redirect(url_for("index"))

    return render_template("signup.html", form=form)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.get_by_username(username)
        if user and user.check_password(password):
            login_user(user)
            next_url = request.args.get("next")
            return redirect(next_url or url_for("index"))

        flash("Invalid username or password.")

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))

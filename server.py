from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
DB_PATH = ROOT / "pokedex.db"
POKEAPI = "https://pokeapi.co/api/v2"

app = FastAPI(title="Pokédex", version="3.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class PokemonAction(BaseModel):
    pokemon_id: int


def db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with db() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS pokemon_cache (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS favorites (
                pokemon_id INTEGER PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS team (
                pokemon_id INTEGER PRIMARY KEY,
                slot INTEGER NOT NULL UNIQUE CHECK(slot BETWEEN 1 AND 6),
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


@app.on_event("startup")
def startup() -> None:
    init_db()


def normalize(data: dict[str, Any]) -> dict[str, Any]:
    stats = {item["stat"]["name"]: item["base_stat"] for item in data["stats"]}
    types = [item["type"]["name"] for item in data["types"]]
    return {
        "id": data["id"],
        "name": data["name"],
        "height": data["height"],
        "weight": data["weight"],
        "types": types,
        "stats": stats,
        "sprite": data["sprites"]["other"]["official-artwork"]["front_default"]
        or data["sprites"]["front_default"],
        "shiny_sprite": data["sprites"]["other"]["official-artwork"]["front_shiny"],
    }


async def fetch_pokemon(identifier: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=12) as client:
        response = await client.get(f"{POKEAPI}/pokemon/{identifier.lower()}")
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Pokémon não encontrado")
    response.raise_for_status()
    return normalize(response.json())


async def get_pokemon(identifier: str) -> dict[str, Any]:
    with db() as connection:
        row = connection.execute(
            "SELECT payload FROM pokemon_cache WHERE name = ? OR id = ?",
            (identifier.lower(), int(identifier) if identifier.isdigit() else -1),
        ).fetchone()
    if row:
        return json.loads(row["payload"])
    pokemon = await fetch_pokemon(identifier)
    with db() as connection:
        connection.execute(
            "INSERT OR REPLACE INTO pokemon_cache (id, name, payload) VALUES (?, ?, ?)",
            (pokemon["id"], pokemon["name"], json.dumps(pokemon)),
        )
    return pokemon


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "pokedex-api"}


@app.get("/api/pokemon")
async def pokemon_list(
    search: str = Query("", max_length=40),
    type: str = Query("", max_length=20),
    limit: int = Query(1025, ge=1, le=1025),
) -> dict[str, Any]:
    if type.strip():
        async with httpx.AsyncClient(timeout=12) as client:
            response = await client.get(f"{POKEAPI}/type/{type.lower()}")
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Tipo não encontrado")
        response.raise_for_status()
        names = [entry["pokemon"]["name"] for entry in response.json()["pokemon"]][:limit]
        return {"items": [await get_pokemon(name) for name in names], "count": len(names)}
    if search.strip():
        async with httpx.AsyncClient(timeout=12) as client:
            response = await client.get(f"{POKEAPI}/pokemon", params={"limit": 1025})
        response.raise_for_status()
        results = [
            item for item in response.json()["results"] if search.lower() in item["name"]
        ][:limit]
        items = [await get_pokemon(item["name"]) for item in results]
        return {"items": items, "count": len(items)}

    async with httpx.AsyncClient(timeout=12) as client:
        response = await client.get(f"{POKEAPI}/pokemon", params={"limit": limit})
    response.raise_for_status()
    items = []
    for index, item in enumerate(response.json()["results"], start=1):
        items.append({
            "id": index,
            "name": item["name"],
            "types": [],
            "stats": {},
            "sprite": f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{index}.png",
            "shiny_sprite": f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/shiny/{index}.png",
        })
    return {"items": items, "count": len(items)}


@app.get("/api/pokemon/{identifier}")
async def pokemon_detail(identifier: str) -> dict[str, Any]:
    return await get_pokemon(identifier)


def ids_for(table: str) -> list[int]:
    with db() as connection:
        rows = connection.execute(f"SELECT pokemon_id FROM {table} ORDER BY pokemon_id").fetchall()
    return [row["pokemon_id"] for row in rows]


@app.get("/api/favorites")
async def favorites() -> list[dict[str, Any]]:
    return [await get_pokemon(str(identifier)) for identifier in ids_for("favorites")]


@app.post("/api/favorites", status_code=201)
def add_favorite(action: PokemonAction) -> dict[str, bool]:
    with db() as connection:
        connection.execute("INSERT OR IGNORE INTO favorites (pokemon_id) VALUES (?)", (action.pokemon_id,))
    return {"saved": True}


@app.delete("/api/favorites/{pokemon_id}")
def remove_favorite(pokemon_id: int) -> dict[str, bool]:
    with db() as connection:
        connection.execute("DELETE FROM favorites WHERE pokemon_id = ?", (pokemon_id,))
    return {"saved": False}


@app.get("/api/team")
async def team() -> list[dict[str, Any]]:
    with db() as connection:
        rows = connection.execute("SELECT pokemon_id, slot FROM team ORDER BY slot").fetchall()
    result = []
    for row in rows:
        pokemon = await get_pokemon(str(row["pokemon_id"]))
        pokemon["slot"] = row["slot"]
        result.append(pokemon)
    return result


@app.post("/api/team", status_code=201)
def add_to_team(action: PokemonAction) -> dict[str, Any]:
    with db() as connection:
        existing = connection.execute("SELECT slot FROM team WHERE pokemon_id = ?", (action.pokemon_id,)).fetchone()
        if existing:
            return {"slot": existing["slot"]}
        used = {row["slot"] for row in connection.execute("SELECT slot FROM team")}
        slot = next((number for number in range(1, 7) if number not in used), None)
        if slot is None:
            raise HTTPException(status_code=409, detail="Seu time já tem seis Pokémon")
        connection.execute("INSERT INTO team (pokemon_id, slot) VALUES (?, ?)", (action.pokemon_id, slot))
    return {"slot": slot}


@app.delete("/api/team/{pokemon_id}")
def remove_from_team(pokemon_id: int) -> dict[str, bool]:
    with db() as connection:
        connection.execute("DELETE FROM team WHERE pokemon_id = ?", (pokemon_id,))
    return {"saved": False}

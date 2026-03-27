import requests
import json
import os
from typing import Optional

# ==================== DADOS LOCAIS ====================
SAVE_FILE = "pokedex_save.json"

DEFAULT_POKEDEX = {
    "Delphox": {"type": "Fire/Psychic", "hp": 75, "region": "Kalos"},
    "Greninja": {"type": "Water/Dark", "hp": 72, "region": "Kalos"},
    "Infernape": {"type": "Fire/Fighting", "hp": 76, "region": "Sinnoh"}
}

# ==================== PERSISTÊNCIA ====================
def carregar_pokedex() -> dict:
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print("[AVISO] Arquivo corrompido. Usando dados padrão.")
    return DEFAULT_POKEDEX.copy()

def salvar_pokedex(pokedex: dict) -> None:
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(pokedex, f, ensure_ascii=False, indent=2)

# ==================== API ====================
BASE_URL = "https://pokeapi.co/api/v2"

def buscar_pokemon_api(nome_ou_numero: str) -> Optional[dict]:
    """Busca dados completos de um Pokémon na PokéAPI."""
    url = f"{BASE_URL}/pokemon/{nome_ou_numero.lower().strip()}"
    try:
        resposta = requests.get(url, timeout=10)
        if resposta.status_code == 404:
            return None
        resposta.raise_for_status()
        return resposta.json()
    except requests.exceptions.ConnectionError:
        print("[ERRO] Sem conexão com a internet.")
        return None
    except requests.exceptions.Timeout:
        print("[ERRO] Tempo esgotado. Tente novamente.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[ERRO] Falha na requisição: {e}")
        return None

def buscar_especie(pokemon_id: int) -> Optional[dict]:
    """Busca dados da espécie (descrição em PT/EN, taxa de captura, etc.)."""
    url = f"{BASE_URL}/pokemon-species/{pokemon_id}"
    try:
        res = requests.get(url, timeout=10)
        if res.ok:
            return res.json()
    except requests.exceptions.RequestException:
        pass
    return None

def buscar_evolucoes(url_chain: str) -> list[str]:
    """Extrai a cadeia de evolução de um Pokémon."""
    try:
        res = requests.get(url_chain, timeout=10)
        if not res.ok:
            return []
        chain = res.json()["chain"]
        evolucoes = []
        node = chain
        while node:
            evolucoes.append(node["species"]["name"].capitalize())
            node = node["evolves_to"][0] if node["evolves_to"] else None
        return evolucoes
    except Exception:
        return []

# ==================== EXIBIÇÃO ====================
SEPARADOR = "=" * 50
SEPARADOR_FINO = "-" * 50

def exibir_pokemon_api(dados: dict, mostrar_movimentos: bool = False) -> None:
    especie = buscar_especie(dados["id"])
    tipos = [t["type"]["name"].capitalize() for t in dados["types"]]
    habilidades = [
        f"{h['ability']['name'].replace('-', ' ').capitalize()}"
        + (" (Oculta)" if h["is_hidden"] else "")
        for h in dados["abilities"]
    ]
    stats = {s["stat"]["name"]: s["base_stat"] for s in dados["stats"]}

    print(f"\n{SEPARADOR}")
    print(f"  #{str(dados['id']).zfill(3)} — {dados['name'].capitalize()}")
    print(SEPARADOR)
    print(f"  Tipo(s)    : {' / '.join(tipos)}")
    print(f"  Altura     : {dados['height'] / 10:.1f} m")
    print(f"  Peso       : {dados['weight'] / 10:.1f} kg")
    print(f"  Exp. base  : {dados.get('base_experience', '—')}")
    print(SEPARADOR_FINO)
    print("  ESTATÍSTICAS BASE:")
    stat_nomes = {
        "hp": "HP", "attack": "Ataque", "defense": "Defesa",
        "special-attack": "Sp. Ataque", "special-defense": "Sp. Defesa", "speed": "Velocidade"
    }
    total = 0
    for chave, nome in stat_nomes.items():
        val = stats.get(chave, 0)
        total += val
        barra = "█" * (val // 10) + "░" * (25 - val // 10)
        print(f"  {nome:<13}: {str(val).rjust(3)} {barra}")
    print(f"  {'TOTAL':<13}: {total}")
    print(SEPARADOR_FINO)
    print(f"  HABILIDADES:")
    for h in habilidades:
        print(f"    • {h}")

    if especie:
        descricao = next(
            (f["flavor_text"].replace("\n", " ").replace("\f", " ")
             for f in especie["flavor_text_entries"]
             if f["language"]["name"] in ("pt", "en")),
            None
        )
        genero = especie.get("gender_rate", -1)
        taxa_captura = especie.get("capture_rate", "—")
        felicidade = especie.get("base_happiness", "—")
        lendario = "✓" if especie.get("is_legendary") else "✗"
        mitico = "✓" if especie.get("is_mythical") else "✗"

        print(SEPARADOR_FINO)
        if descricao:
            print(f"  DESCRIÇÃO: {descricao}")
        print(f"  Taxa de captura : {taxa_captura}")
        print(f"  Felicidade base : {felicidade}")
        print(f"  Lendário        : {lendario}  |  Mítico: {mitico}")

        if genero == -1:
            print("  Gênero          : Sem gênero")
        else:
            pct_fem = genero / 8 * 100
            print(f"  Gênero          : ♀ {pct_fem:.0f}%  ♂ {100 - pct_fem:.0f}%")

        if especie.get("evolution_chain"):
            evolucoes = buscar_evolucoes(especie["evolution_chain"]["url"])
            if len(evolucoes) > 1:
                print(f"  Evolução        : {' → '.join(evolucoes)}")

    if mostrar_movimentos:
        movimentos = [m["move"]["name"].replace("-", " ").capitalize() for m in dados["moves"][:20]]
        print(SEPARADOR_FINO)
        print("  MOVIMENTOS (primeiros 20):")
        for i, m in enumerate(movimentos, 1):
            print(f"    {str(i).rjust(2)}. {m}")

    print(SEPARADOR)

def exibir_pokemon_local(nome: str, dados: dict) -> None:
    print(f"\n{SEPARADOR}")
    print(f"  {nome} (local)")
    print(SEPARADOR)
    print(f"  Tipo   : {dados['type']}")
    print(f"  HP     : {dados['hp']}")
    print(f"  Região : {dados.get('region', '—')}")
    print(SEPARADOR)

def listar_pokedex(pokedex: dict, filtro_tipo: str = "") -> None:
    entradas = [
        (nome, dados) for nome, dados in pokedex.items()
        if not filtro_tipo or filtro_tipo.lower() in dados["type"].lower()
    ]
    if not entradas:
        print("\n  Nenhum Pokémon encontrado.")
        return
    print(f"\n{SEPARADOR}")
    print(f"  POKÉDEX LOCAL ({len(entradas)} pokémon)")
    print(SEPARADOR)
    for i, (nome, dados) in enumerate(entradas, 1):
        print(f"  {str(i).rjust(2)}. {nome:<15} | {dados['type']:<20} | HP: {dados['hp']} | {dados.get('region','—')}")
    print(SEPARADOR)

# ==================== MENU ====================
def menu_adicionar(pokedex: dict) -> None:
    print("\n  Como deseja adicionar?")
    print("  1 - Manual")
    print("  2 - Importar da API")
    opcao = input("  Escolha: ").strip()

    if opcao == "1":
        nome = input("  Nome: ").strip().capitalize()
        if not nome:
            print("  Nome inválido."); return
        if nome in pokedex:
            print(f"  {nome} já está na Pokédex."); return
        tipo = input("  Tipo (ex: Fire/Flying): ").strip()
        try:
            hp = int(input("  HP: ").strip())
        except ValueError:
            print("  HP inválido."); return
        region = input("  Região: ").strip()
        pokedex[nome] = {"type": tipo, "hp": hp, "region": region}
        salvar_pokedex(pokedex)
        print(f"\n  ✓ {nome} adicionado com sucesso!")

    elif opcao == "2":
        query = input("  Nome ou número do Pokémon: ").strip()
        dados = buscar_pokemon_api(query)
        if not dados:
            print("  Pokémon não encontrado na API."); return
        nome = dados["name"].capitalize()
        tipos = "/".join(t["type"]["name"].capitalize() for t in dados["types"])
        hp = next((s["base_stat"] for s in dados["stats"] if s["stat"]["name"] == "hp"), 0)
        pokedex[nome] = {
            "type": tipos,
            "hp": hp,
            "region": input(f"  Região de {nome}: ").strip() or "—"
        }
        salvar_pokedex(pokedex)
        print(f"\n  ✓ {nome} importado da API!")
        exibir_pokemon_api(dados)
    else:
        print("  Opção inválida.")

def menu_remover(pokedex: dict) -> None:
    nome = input("  Nome do Pokémon a remover: ").strip().capitalize()
    if nome not in pokedex:
        print(f"  {nome} não encontrado."); return
    confirm = input(f"  Remover {nome}? (s/n): ").strip().lower()
    if confirm == 's':
        del pokedex[nome]
        salvar_pokedex(pokedex)
        print(f"  ✓ {nome} removido.")

def menu_comparar(pokedex: dict) -> None:
    print("\n  Comparar dois Pokémons via API")
    a = input("  Pokémon 1: ").strip()
    b = input("  Pokémon 2: ").strip()
    dados_a = buscar_pokemon_api(a)
    dados_b = buscar_pokemon_api(b)
    if not dados_a or not dados_b:
        print("  Um ou ambos não encontrados."); return

    stats_a = {s["stat"]["name"]: s["base_stat"] for s in dados_a["stats"]}
    stats_b = {s["stat"]["name"]: s["base_stat"] for s in dados_b["stats"]}
    stat_nomes = {
        "hp": "HP", "attack": "Ataque", "defense": "Defesa",
        "special-attack": "Sp. Atk", "special-defense": "Sp. Def", "speed": "Velocidade"
    }

    nome_a = dados_a["name"].capitalize()
    nome_b = dados_b["name"].capitalize()
    print(f"\n{SEPARADOR}")
    print(f"  {nome_a:<20} vs  {nome_b}")
    print(SEPARADOR)
    vitorias_a = vitorias_b = 0
    for chave, nome in stat_nomes.items():
        va, vb = stats_a.get(chave, 0), stats_b.get(chave, 0)
        vencedor = f"◄ {nome_a}" if va > vb else (f"► {nome_b}" if vb > va else "EMPATE")
        print(f"  {nome:<13}: {str(va).rjust(3)}  vs  {str(vb).ljust(3)} {vencedor}")
        if va > vb: vitorias_a += 1
        elif vb > va: vitorias_b += 1
    print(SEPARADOR)
    total_a = sum(stats_a.values())
    total_b = sum(stats_b.values())
    print(f"  TOTAL: {nome_a} {total_a} | {nome_b} {total_b}")
    print(f"  VENCEDOR GERAL: {'EMPATE' if total_a == total_b else (nome_a if total_a > total_b else nome_b)}")
    print(SEPARADOR)

# ==================== LOOP PRINCIPAL ====================
def main():
    pokedex = carregar_pokedex()
    print("\n  ╔══════════════════════════╗")
    print("  ║        POKÉDEX v2        ║")
    print("  ╚══════════════════════════╝")

    while True:
        print(f"\n  [Pokémons: {len(pokedex)}]")
        print("  1 - Adicionar Pokémon")
        print("  2 - Buscar local")
        print("  3 - Buscar na API (detalhes completos)")
        print("  4 - Listar todos")
        print("  5 - Filtrar por tipo")
        print("  6 - Comparar dois Pokémons")
        print("  7 - Remover Pokémon")
        print("  8 - Sair")

        opcao = input("\n  Escolha: ").strip()

        if opcao == "1":
            menu_adicionar(pokedex)

        elif opcao == "2":
            nome = input("  Nome: ").strip().capitalize()
            if nome in pokedex:
                exibir_pokemon_local(nome, pokedex[nome])
            else:
                print(f"  {nome} não encontrado na lista local.")

        elif opcao == "3":
            query = input("  Nome ou número: ").strip()
            dados = buscar_pokemon_api(query)
            if dados:
                ver_movimentos = input("  Ver movimentos? (s/n): ").strip().lower() == 's'
                exibir_pokemon_api(dados, ver_movimentos)
            else:
                print(f"  Pokémon não encontrado.")

        elif opcao == "4":
            listar_pokedex(pokedex)

        elif opcao == "5":
            filtro = input("  Tipo (ex: Fire, Water): ").strip()
            listar_pokedex(pokedex, filtro)

        elif opcao == "6":
            menu_comparar(pokedex)

        elif opcao == "7":
            menu_remover(pokedex)

        elif opcao == "8":
            salvar_pokedex(pokedex)
            print("\n  Pokédex salva. Até logo, Treinador!\n")
            break

        else:
            print("  Opção inválida.")

if __name__ == "__main__":
    main()

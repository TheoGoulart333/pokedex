import requests

pokedex = {
    "Delphox": {
        "type": "Fire/Psychic",
        "hp": 75,
        "region": "Kalos"
    },
    "Greninja": {
        "type": "Water/Dark",
        "hp": 72,
        "region": "Kalos"
    },
    "Infernape": {
        "type": "Fire/Fighting",
        "hp": 76,
        "region": "Sinnoh"
    }
}


def buscar_pokemon(nome):
    url = f"https://pokeapi.co/api/v2/pokemon/{nome.lower()}"
    resposta = requests.get(url)

    if resposta.status_code != 200:
        print("Pokémon não encontrado")
        return

    dados = resposta.json()

    nome = dados["name"].capitalize()
    altura = dados["height"] / 10
    peso = dados["weight"] / 10

    tipos = [t["type"]["name"] for t in dados["types"]]
    habilidades = [h["ability"]["name"] for h in dados["abilities"]]

    print("\n====== Pokedex ======")
    print(f"Nome: {nome}")
    print(f"Altura: {altura} m")
    print(f"Peso: {peso} kg")
    print(f"Tipo: {', '.join(tipos)}")
    print(f"Habilidades: {', '.join(habilidades)}")
    print("====================")


while True:
    print("\n1 - Adicionar Pokémon")
    print("2 - Buscar Pokémon (local)")
    print("3 - Buscar Pokémon na API")
    print("4 - Sair")

    opcao = input("Escolha uma opção: ")

    if opcao == "1":
        nome = input("Digite o nome do Pokémon: ")
        tipo = input("Tipo: ")
        hp = int(input("HP: "))
        region = input("Região: ")

        pokedex[nome] = {
            "type": tipo,
            "hp": hp,
            "region": region
        }

        print("Pokémon adicionado!")

    elif opcao == "2":
        nome = input("Digite o nome do Pokémon: ")

        if nome in pokedex:
            print("Tipo:", pokedex[nome]["type"])
            print("HP:", pokedex[nome]["hp"])
            print("Região:", pokedex[nome]["region"])
        else:
            print("Pokémon não encontrado")

    elif opcao == "3":
        nome = input("Digite o nome do Pokémon: ")
        buscar_pokemon(nome)

    elif opcao == "4":
        print("Saindo da Pokedex...")
        break

    else:
        print("Opção inválida")

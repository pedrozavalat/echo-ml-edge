from libs import GoogleDriveClient, SnapshotClient
import os
from dotenv import load_dotenv
import asyncio
import json

load_dotenv()
REGIONES = [15, 16]
RETRIEVE_URLS = True

def save_json(data, filename):
    with open(os.path.join("json", filename), "w") as f:
        json.dump(data, f, indent=4)

def load_json(filename):
    with open(os.path.join("json", filename), "r") as f:
        return json.load(f)

async def main():
    sp_client = SnapshotClient()
    gd_client = GoogleDriveClient(api_key=os.getenv("GOOGLE_API_KEY"))
    regiones = await sp_client.regiones()
    regiones_filtradas = [r for r in regiones["regiones"] if r["codigo"] in REGIONES]

    tabla = {}
    if RETRIEVE_URLS:
        for region in regiones_filtradas:
            codigo_region = region["codigo"]
            tabla[codigo_region] = {}
            unidades = await sp_client.unidades_by_region(codigo_region)
            for unidad in unidades["unidades"]:
                codigo_unidad = unidad["codigo"]
                tabla[codigo_region][codigo_unidad] = {}
                years = await sp_client.year_by_unidad(codigo_unidad)
                for year in years["years"]:
                    year_value = year["year"]
                    tabla[codigo_region][codigo_unidad][year_value] = {}
                    especies = await sp_client.especie_by_unidadyear(
                        codigo_unidad, year_value
                    )
                    for especie in especies["especies"]:
                        codigo_especie = especie["codigo"]
                        urls = await sp_client.grillas_by_year_especie_geojson(
                            codigo_unidad, year_value, codigo_especie, only_urls=True
                        )
                        tabla[codigo_region][codigo_unidad][year_value][codigo_especie] = (
                            urls
                        )
        save_json(tabla, "tabla.json")
    else:
        tabla = load_json("tabla.json")
        


if __name__ == "__main__":
    asyncio.run(main())

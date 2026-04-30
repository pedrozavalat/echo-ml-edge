from libs import SnapshotClient
from dotenv import load_dotenv
import asyncio

load_dotenv()


async def main():
    async with SnapshotClient() as snapshot_client:
        regiones = await snapshot_client.regiones()
        print(regiones)
        first_region_code = regiones["regiones"][0]["codigo"]
        unidades = await snapshot_client.unidades_by_region(first_region_code)
        print(unidades)
        first_unidad_code = unidades["unidades"][0]["codigo"]
        years_by_unidad = await snapshot_client.year_by_unidad(
            codigo_unidad=first_unidad_code
        )
        print(years_by_unidad)
        first_year = years_by_unidad["years"][0]["year"]
        especies_by_unidad_year = await snapshot_client.especie_by_unidadyear(
            first_unidad_code, first_year
        )
        print(especies_by_unidad_year)
        first_especie_code = especies_by_unidad_year["especies"][0]["codigo"]
        folders = await snapshot_client.grillas_by_year_especie_geojson(
            first_unidad_code, first_year, first_especie_code, only_urls=True
        )
        print(folders)


if __name__ == "__main__":
    asyncio.run(main())

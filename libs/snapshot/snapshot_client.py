import aiohttp
import certifi
import ssl
from yarl import URL


class SnapshotClient:
    def __init__(self) -> None:
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.connector = aiohttp.TCPConnector(ssl=self.ssl_context)
        self.session: aiohttp.ClientSession | None = None
        self.url = "https://app.fotomonitoreo.cl/"
        self.origin = "https://app.fotomonitoreo.cl"
        self.headers = None

    async def __aenter__(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(connector=self.connector)
        await self._get_headers()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session is not None and not self.session.closed:
            await self.session.close()
        if not self.connector.closed:
            await self.connector.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(connector=self.connector)
        return self.session

    async def _get_headers(self) -> dict:
        session = await self._ensure_session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": self.url,
            "Origin": self.origin,
            "X-Requested-With": "XMLHttpRequest",
        }
        async with session.get(self.url, headers=headers) as response:
            await response.text()
        csrf_token = session.cookie_jar.filter_cookies(URL(self.url)).get("csrftoken")
        csrf_value = csrf_token.value if csrf_token is not None else None
        headers = {
            **headers,
            "X-CSRFToken": csrf_value,
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        }
        self.headers = headers
        return headers

    async def _request_json(self, method: str, path: str, data: dict | None = None):
        session = await self._ensure_session()
        if self.headers is None:
            await self._get_headers()

        async with session.request(
            method,
            f"{self.url}{path}",
            headers=self.headers,
            data=data,
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def regiones(self) -> dict:
        try:
            return await self._request_json("GET", "visor/regiones/")
        except aiohttp.ClientError as e:
            print(f"Error fetching regions: {e}")
            return {}

    async def unidades_by_region(self, codigo_region: int) -> dict:
        try:
            data = {
                "codigo_region": int(codigo_region),
            }
            return await self._request_json("POST", "visor/unidades_by_region/", data)
        except aiohttp.ClientError as e:
            print(f"Error fetching unidades for region {codigo_region}: {e}")
            return {}

    async def year_by_unidad(self, codigo_unidad: str):
        try:
            data = {
                "codigo_unidad": codigo_unidad,
            }
            return await self._request_json("POST", "visor/year_by_unidad/", data)
        except aiohttp.ClientError as e:
            print(f"Error fetching years for unidad {codigo_unidad}: {e}")
            return {}

    async def especie_by_unidadyear(self, codigo_unidad: str, year: int):
        try:
            data = {
                "codigo_unidad": codigo_unidad,
                "year": year,
            }
            return await self._request_json(
                "POST", "visor/especie_by_unidadyear/", data
            )
        except aiohttp.ClientError as e:
            print(
                f"Error fetching especies for unidad {codigo_unidad} and year {year}: {e}"
            )
            return {}

    async def grillas_by_year_geojson(self, codigo_unidad: str, year: int):
        try:
            data = {
                "codigo_unidad": codigo_unidad,
                "year": year,
            }
            return await self._request_json(
                "POST", "visor/grillas_by_year_geojson/", data
            )
        except aiohttp.ClientError as e:
            print(
                f"Error fetching grillas for unidad {codigo_unidad} and year {year}: {e}"
            )
            return {}

    async def grillas_by_year_especie_geojson(
        self, codigo_unidad: str, year: int, especie: str, only_urls: bool = False
    ):
        try:
            data = {
                "codigo_unidad": codigo_unidad,
                "year": year,
                "especie": especie,
            }
            response = await self._request_json(
                "POST", "visor/grillas_by_year_especie_geojson/", data
            )
            if not only_urls:
                return response
            return list(
                map(lambda f: f["properties"]["gdrive_url"], response["features"])
            )
        except aiohttp.ClientError as e:
            print(
                f"Error fetching grillas for unidad {codigo_unidad}, year {year} and especie {especie}: {e}"
            )
            return {}

import asyncio
from tempfile import NamedTemporaryFile

import twint
from aiohttp.client_exceptions import ClientConnectionError


def _get_twint_config(limit):
    c = twint.Config()
    c.Limit = limit
    c.Hide_output = True
    return c


def get_followers_config(username, limit):
    c = _get_twint_config(limit)
    c.Username = username
    c.Followers = True
    return c


def get_search_config(query, limit):
    c = _get_twint_config(limit)
    c.TwitterSearch = True
    c.Search = query
    return c


def get_user_timeline(username, limit):
    c = _get_twint_config(limit)
    c.Username = username
    c.Profile = True
    c.Retweets = True
    return c


async def _single_fetch_using_twint(twint_config, retries, callback=None):
    with NamedTemporaryFile("r+", delete=True) as tmp_file:
        twint_config.Output = tmp_file.name
        for essai in range(1, retries + 1):
            try:
                await twint.run.Twint(twint_config).main()
            except ClientConnectionError as e:
                print(
                    f"failed to fetch using twing config {vars(twint_config)}, try number {essai}. \nError: {e}"
                )
            except Exception as e:
                print(
                    f"Unknown error {e} when fetching using twint config {vars(twint_config)}, try number {essai}"
                )
            if essai == retries:
                print(
                    f"That was the last try to fetch using twint config {vars(twint_config)}"
                )
            else:
                contents = tmp_file.read()
                if callback is not None:
                    callback(twint_config.Username, contents)
                return contents


async def _semaphored_fetch_using_twint(
    semaphore, twint_config, retries, callback=None
):
    async with semaphore:
        return await _single_fetch_using_twint(twint_config, retries, callback=callback)


def fetch_using_twint(twint_configs, retries, threads=None, callback=None):
    fetching_loop = asyncio.get_event_loop()
    if not threads:
        results = [
            fetching_loop.run_until_complete(
                _single_fetch_using_twint(config, retries, callback=callback)
            )
            for config in twint_configs
        ]
    else:
        semaphore = asyncio.Semaphore(threads)
        concurrent_fetching_task = asyncio.gather(
            *[
                _semaphored_fetch_using_twint(
                    semaphore, config, retries, callback=callback
                )
                for config in twint_configs
            ]
        )
        results = fetching_loop.run_until_complete(concurrent_fetching_task)
    return results

# pylint: disable=too-many-lines
# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
from typing import Any, AsyncIterable, Callable, Dict, IO, Optional, TypeVar, Union, overload
from urllib.parse import parse_qs, urljoin, urlparse

from azure.core.async_paging import AsyncItemPaged, AsyncList
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
    ResourceNotModifiedError,
    map_error,
)
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import AsyncHttpResponse
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.core.tracing.decorator_async import distributed_trace_async
from azure.core.utils import case_insensitive_dict
from azure.mgmt.core.exceptions import ARMErrorFormat

from ... import models as _models
from ..._vendor import _convert_request
from ...operations._iot_security_solution_operations import (
    build_create_or_update_request,
    build_delete_request,
    build_get_request,
    build_list_by_resource_group_request,
    build_list_by_subscription_request,
    build_update_request,
)

T = TypeVar("T")
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, AsyncHttpResponse], T, Dict[str, Any]], Any]]


class IotSecuritySolutionOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.security.v2019_08_01.aio.SecurityCenter`'s
        :attr:`iot_security_solution` attribute.
    """

    models = _models

    def __init__(self, *args, **kwargs) -> None:
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop("client")
        self._config = input_args.pop(0) if input_args else kwargs.pop("config")
        self._serialize = input_args.pop(0) if input_args else kwargs.pop("serializer")
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop("deserializer")

    @distributed_trace
    def list_by_subscription(
        self, filter: Optional[str] = None, **kwargs: Any
    ) -> AsyncIterable["_models.IoTSecuritySolutionModel"]:
        """Use this method to get the list of IoT Security solutions by subscription.

        :param filter: Filter the IoT Security solution with OData syntax. Supports filtering by
         iotHubs. Default value is None.
        :type filter: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: An iterator like instance of either IoTSecuritySolutionModel or the result of
         cls(response)
        :rtype:
         ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.security.v2019_08_01.models.IoTSecuritySolutionModel]
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version = kwargs.pop("api_version", _params.pop("api-version", "2019-08-01"))  # type: str
        cls = kwargs.pop("cls", None)  # type: ClsType[_models.IoTSecuritySolutionsList]

        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        def prepare_request(next_link=None):
            if not next_link:

                request = build_list_by_subscription_request(
                    subscription_id=self._config.subscription_id,
                    filter=filter,
                    api_version=api_version,
                    template_url=self.list_by_subscription.metadata["url"],
                    headers=_headers,
                    params=_params,
                )
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)  # type: ignore

            else:
                # make call to next link with the client's api-version
                _parsed_next_link = urlparse(next_link)
                _next_request_params = case_insensitive_dict(parse_qs(_parsed_next_link.query))
                _next_request_params["api-version"] = self._config.api_version
                request = HttpRequest("GET", urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)  # type: ignore
                request.method = "GET"
            return request

        async def extract_data(pipeline_response):
            deserialized = self._deserialize("IoTSecuritySolutionsList", pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return deserialized.next_link or None, AsyncList(list_of_elem)

        async def get_next(next_link=None):
            request = prepare_request(next_link)

            pipeline_response = await self._client._pipeline.run(  # type: ignore # pylint: disable=protected-access
                request, stream=False, **kwargs
            )
            response = pipeline_response.http_response

            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)

            return pipeline_response

        return AsyncItemPaged(get_next, extract_data)

    list_by_subscription.metadata = {"url": "/subscriptions/{subscriptionId}/providers/Microsoft.Security/iotSecuritySolutions"}  # type: ignore

    @distributed_trace
    def list_by_resource_group(
        self, resource_group_name: str, filter: Optional[str] = None, **kwargs: Any
    ) -> AsyncIterable["_models.IoTSecuritySolutionModel"]:
        """Use this method to get the list IoT Security solutions organized by resource group.

        :param resource_group_name: The name of the resource group within the user's subscription. The
         name is case insensitive. Required.
        :type resource_group_name: str
        :param filter: Filter the IoT Security solution with OData syntax. Supports filtering by
         iotHubs. Default value is None.
        :type filter: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: An iterator like instance of either IoTSecuritySolutionModel or the result of
         cls(response)
        :rtype:
         ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.security.v2019_08_01.models.IoTSecuritySolutionModel]
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version = kwargs.pop("api_version", _params.pop("api-version", "2019-08-01"))  # type: str
        cls = kwargs.pop("cls", None)  # type: ClsType[_models.IoTSecuritySolutionsList]

        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        def prepare_request(next_link=None):
            if not next_link:

                request = build_list_by_resource_group_request(
                    resource_group_name=resource_group_name,
                    subscription_id=self._config.subscription_id,
                    filter=filter,
                    api_version=api_version,
                    template_url=self.list_by_resource_group.metadata["url"],
                    headers=_headers,
                    params=_params,
                )
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)  # type: ignore

            else:
                # make call to next link with the client's api-version
                _parsed_next_link = urlparse(next_link)
                _next_request_params = case_insensitive_dict(parse_qs(_parsed_next_link.query))
                _next_request_params["api-version"] = self._config.api_version
                request = HttpRequest("GET", urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)  # type: ignore
                request.method = "GET"
            return request

        async def extract_data(pipeline_response):
            deserialized = self._deserialize("IoTSecuritySolutionsList", pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return deserialized.next_link or None, AsyncList(list_of_elem)

        async def get_next(next_link=None):
            request = prepare_request(next_link)

            pipeline_response = await self._client._pipeline.run(  # type: ignore # pylint: disable=protected-access
                request, stream=False, **kwargs
            )
            response = pipeline_response.http_response

            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)

            return pipeline_response

        return AsyncItemPaged(get_next, extract_data)

    list_by_resource_group.metadata = {"url": "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Security/iotSecuritySolutions"}  # type: ignore

    @distributed_trace_async
    async def get(
        self, resource_group_name: str, solution_name: str, **kwargs: Any
    ) -> _models.IoTSecuritySolutionModel:
        """User this method to get details of a specific IoT Security solution based on solution name.

        :param resource_group_name: The name of the resource group within the user's subscription. The
         name is case insensitive. Required.
        :type resource_group_name: str
        :param solution_name: The name of the IoT Security solution. Required.
        :type solution_name: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: IoTSecuritySolutionModel or the result of cls(response)
        :rtype: ~azure.mgmt.security.v2019_08_01.models.IoTSecuritySolutionModel
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version = kwargs.pop("api_version", _params.pop("api-version", "2019-08-01"))  # type: str
        cls = kwargs.pop("cls", None)  # type: ClsType[_models.IoTSecuritySolutionModel]

        request = build_get_request(
            resource_group_name=resource_group_name,
            solution_name=solution_name,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            template_url=self.get.metadata["url"],
            headers=_headers,
            params=_params,
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)  # type: ignore

        pipeline_response = await self._client._pipeline.run(  # type: ignore # pylint: disable=protected-access
            request, stream=False, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        deserialized = self._deserialize("IoTSecuritySolutionModel", pipeline_response)

        if cls:
            return cls(pipeline_response, deserialized, {})

        return deserialized

    get.metadata = {"url": "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Security/iotSecuritySolutions/{solutionName}"}  # type: ignore

    @overload
    async def create_or_update(
        self,
        resource_group_name: str,
        solution_name: str,
        iot_security_solution_data: _models.IoTSecuritySolutionModel,
        *,
        content_type: str = "application/json",
        **kwargs: Any
    ) -> _models.IoTSecuritySolutionModel:
        """Use this method to create or update yours IoT Security solution.

        :param resource_group_name: The name of the resource group within the user's subscription. The
         name is case insensitive. Required.
        :type resource_group_name: str
        :param solution_name: The name of the IoT Security solution. Required.
        :type solution_name: str
        :param iot_security_solution_data: The security solution data. Required.
        :type iot_security_solution_data:
         ~azure.mgmt.security.v2019_08_01.models.IoTSecuritySolutionModel
        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.
         Default value is "application/json".
        :paramtype content_type: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: IoTSecuritySolutionModel or the result of cls(response)
        :rtype: ~azure.mgmt.security.v2019_08_01.models.IoTSecuritySolutionModel
        :raises ~azure.core.exceptions.HttpResponseError:
        """

    @overload
    async def create_or_update(
        self,
        resource_group_name: str,
        solution_name: str,
        iot_security_solution_data: IO,
        *,
        content_type: str = "application/json",
        **kwargs: Any
    ) -> _models.IoTSecuritySolutionModel:
        """Use this method to create or update yours IoT Security solution.

        :param resource_group_name: The name of the resource group within the user's subscription. The
         name is case insensitive. Required.
        :type resource_group_name: str
        :param solution_name: The name of the IoT Security solution. Required.
        :type solution_name: str
        :param iot_security_solution_data: The security solution data. Required.
        :type iot_security_solution_data: IO
        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.
         Default value is "application/json".
        :paramtype content_type: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: IoTSecuritySolutionModel or the result of cls(response)
        :rtype: ~azure.mgmt.security.v2019_08_01.models.IoTSecuritySolutionModel
        :raises ~azure.core.exceptions.HttpResponseError:
        """

    @distributed_trace_async
    async def create_or_update(
        self,
        resource_group_name: str,
        solution_name: str,
        iot_security_solution_data: Union[_models.IoTSecuritySolutionModel, IO],
        **kwargs: Any
    ) -> _models.IoTSecuritySolutionModel:
        """Use this method to create or update yours IoT Security solution.

        :param resource_group_name: The name of the resource group within the user's subscription. The
         name is case insensitive. Required.
        :type resource_group_name: str
        :param solution_name: The name of the IoT Security solution. Required.
        :type solution_name: str
        :param iot_security_solution_data: The security solution data. Is either a model type or a IO
         type. Required.
        :type iot_security_solution_data:
         ~azure.mgmt.security.v2019_08_01.models.IoTSecuritySolutionModel or IO
        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.
         Default value is None.
        :paramtype content_type: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: IoTSecuritySolutionModel or the result of cls(response)
        :rtype: ~azure.mgmt.security.v2019_08_01.models.IoTSecuritySolutionModel
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version = kwargs.pop("api_version", _params.pop("api-version", "2019-08-01"))  # type: str
        content_type = kwargs.pop("content_type", _headers.pop("Content-Type", None))  # type: Optional[str]
        cls = kwargs.pop("cls", None)  # type: ClsType[_models.IoTSecuritySolutionModel]

        content_type = content_type or "application/json"
        _json = None
        _content = None
        if isinstance(iot_security_solution_data, (IO, bytes)):
            _content = iot_security_solution_data
        else:
            _json = self._serialize.body(iot_security_solution_data, "IoTSecuritySolutionModel")

        request = build_create_or_update_request(
            resource_group_name=resource_group_name,
            solution_name=solution_name,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            content_type=content_type,
            json=_json,
            content=_content,
            template_url=self.create_or_update.metadata["url"],
            headers=_headers,
            params=_params,
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)  # type: ignore

        pipeline_response = await self._client._pipeline.run(  # type: ignore # pylint: disable=protected-access
            request, stream=False, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200, 201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        if response.status_code == 200:
            deserialized = self._deserialize("IoTSecuritySolutionModel", pipeline_response)

        if response.status_code == 201:
            deserialized = self._deserialize("IoTSecuritySolutionModel", pipeline_response)

        if cls:
            return cls(pipeline_response, deserialized, {})

        return deserialized

    create_or_update.metadata = {"url": "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Security/iotSecuritySolutions/{solutionName}"}  # type: ignore

    @overload
    async def update(
        self,
        resource_group_name: str,
        solution_name: str,
        update_iot_security_solution_data: _models.UpdateIotSecuritySolutionData,
        *,
        content_type: str = "application/json",
        **kwargs: Any
    ) -> _models.IoTSecuritySolutionModel:
        """Use this method to update existing IoT Security solution tags or user defined resources. To
        update other fields use the CreateOrUpdate method.

        :param resource_group_name: The name of the resource group within the user's subscription. The
         name is case insensitive. Required.
        :type resource_group_name: str
        :param solution_name: The name of the IoT Security solution. Required.
        :type solution_name: str
        :param update_iot_security_solution_data: The security solution data. Required.
        :type update_iot_security_solution_data:
         ~azure.mgmt.security.v2019_08_01.models.UpdateIotSecuritySolutionData
        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.
         Default value is "application/json".
        :paramtype content_type: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: IoTSecuritySolutionModel or the result of cls(response)
        :rtype: ~azure.mgmt.security.v2019_08_01.models.IoTSecuritySolutionModel
        :raises ~azure.core.exceptions.HttpResponseError:
        """

    @overload
    async def update(
        self,
        resource_group_name: str,
        solution_name: str,
        update_iot_security_solution_data: IO,
        *,
        content_type: str = "application/json",
        **kwargs: Any
    ) -> _models.IoTSecuritySolutionModel:
        """Use this method to update existing IoT Security solution tags or user defined resources. To
        update other fields use the CreateOrUpdate method.

        :param resource_group_name: The name of the resource group within the user's subscription. The
         name is case insensitive. Required.
        :type resource_group_name: str
        :param solution_name: The name of the IoT Security solution. Required.
        :type solution_name: str
        :param update_iot_security_solution_data: The security solution data. Required.
        :type update_iot_security_solution_data: IO
        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.
         Default value is "application/json".
        :paramtype content_type: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: IoTSecuritySolutionModel or the result of cls(response)
        :rtype: ~azure.mgmt.security.v2019_08_01.models.IoTSecuritySolutionModel
        :raises ~azure.core.exceptions.HttpResponseError:
        """

    @distributed_trace_async
    async def update(
        self,
        resource_group_name: str,
        solution_name: str,
        update_iot_security_solution_data: Union[_models.UpdateIotSecuritySolutionData, IO],
        **kwargs: Any
    ) -> _models.IoTSecuritySolutionModel:
        """Use this method to update existing IoT Security solution tags or user defined resources. To
        update other fields use the CreateOrUpdate method.

        :param resource_group_name: The name of the resource group within the user's subscription. The
         name is case insensitive. Required.
        :type resource_group_name: str
        :param solution_name: The name of the IoT Security solution. Required.
        :type solution_name: str
        :param update_iot_security_solution_data: The security solution data. Is either a model type or
         a IO type. Required.
        :type update_iot_security_solution_data:
         ~azure.mgmt.security.v2019_08_01.models.UpdateIotSecuritySolutionData or IO
        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.
         Default value is None.
        :paramtype content_type: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: IoTSecuritySolutionModel or the result of cls(response)
        :rtype: ~azure.mgmt.security.v2019_08_01.models.IoTSecuritySolutionModel
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version = kwargs.pop("api_version", _params.pop("api-version", "2019-08-01"))  # type: str
        content_type = kwargs.pop("content_type", _headers.pop("Content-Type", None))  # type: Optional[str]
        cls = kwargs.pop("cls", None)  # type: ClsType[_models.IoTSecuritySolutionModel]

        content_type = content_type or "application/json"
        _json = None
        _content = None
        if isinstance(update_iot_security_solution_data, (IO, bytes)):
            _content = update_iot_security_solution_data
        else:
            _json = self._serialize.body(update_iot_security_solution_data, "UpdateIotSecuritySolutionData")

        request = build_update_request(
            resource_group_name=resource_group_name,
            solution_name=solution_name,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            content_type=content_type,
            json=_json,
            content=_content,
            template_url=self.update.metadata["url"],
            headers=_headers,
            params=_params,
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)  # type: ignore

        pipeline_response = await self._client._pipeline.run(  # type: ignore # pylint: disable=protected-access
            request, stream=False, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        deserialized = self._deserialize("IoTSecuritySolutionModel", pipeline_response)

        if cls:
            return cls(pipeline_response, deserialized, {})

        return deserialized

    update.metadata = {"url": "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Security/iotSecuritySolutions/{solutionName}"}  # type: ignore

    @distributed_trace_async
    async def delete(  # pylint: disable=inconsistent-return-statements
        self, resource_group_name: str, solution_name: str, **kwargs: Any
    ) -> None:
        """Use this method to delete yours IoT Security solution.

        :param resource_group_name: The name of the resource group within the user's subscription. The
         name is case insensitive. Required.
        :type resource_group_name: str
        :param solution_name: The name of the IoT Security solution. Required.
        :type solution_name: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: None or the result of cls(response)
        :rtype: None
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version = kwargs.pop("api_version", _params.pop("api-version", "2019-08-01"))  # type: str
        cls = kwargs.pop("cls", None)  # type: ClsType[None]

        request = build_delete_request(
            resource_group_name=resource_group_name,
            solution_name=solution_name,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            template_url=self.delete.metadata["url"],
            headers=_headers,
            params=_params,
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)  # type: ignore

        pipeline_response = await self._client._pipeline.run(  # type: ignore # pylint: disable=protected-access
            request, stream=False, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        if cls:
            return cls(pipeline_response, None, {})

    delete.metadata = {"url": "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Security/iotSecuritySolutions/{solutionName}"}  # type: ignore

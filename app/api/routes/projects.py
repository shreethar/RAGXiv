from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies import get_current_user_id, get_project_service
from app.api.errors import call_application
from app.api.schemas.papers import PaperResponse
from app.api.schemas.projects import (
    ProjectCreateRequest,
    ProjectListResponse,
    ProjectPaperAddRequest,
    ProjectPapersResponse,
    ProjectResponse,
)
from app.services.project_service import ProjectService


router = APIRouter(prefix="/projects", tags=["projects"])


@router.post(
    "",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_project(
    request: ProjectCreateRequest,
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    service: Annotated[ProjectService, Depends(get_project_service)],
) -> ProjectResponse:
    project = call_application(
        lambda: service.create(user_id=user_id, name=request.name)
    )
    return ProjectResponse.model_validate(project)


@router.get("", response_model=ProjectListResponse)
def list_projects(
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    service: Annotated[ProjectService, Depends(get_project_service)],
) -> ProjectListResponse:
    projects = service.list_projects(user_id=user_id)
    return ProjectListResponse(
        projects=[ProjectResponse.model_validate(item) for item in projects]
    )


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    project_id: UUID,
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    service: Annotated[ProjectService, Depends(get_project_service)],
) -> ProjectResponse:
    project = call_application(
        lambda: service.get(user_id=user_id, project_id=project_id)
    )
    return ProjectResponse.model_validate(project)


@router.get(
    "/{project_id}/papers",
    response_model=ProjectPapersResponse,
)
def list_project_papers(
    project_id: UUID,
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    service: Annotated[ProjectService, Depends(get_project_service)],
) -> ProjectPapersResponse:
    papers = call_application(
        lambda: service.list_papers(
            user_id=user_id,
            project_id=project_id,
        )
    )
    return ProjectPapersResponse(
        papers=[PaperResponse.model_validate(item) for item in papers]
    )


@router.post(
    "/{project_id}/papers",
    response_model=ProjectPapersResponse,
)
def add_project_papers(
    project_id: UUID,
    request: ProjectPaperAddRequest,
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    service: Annotated[ProjectService, Depends(get_project_service)],
) -> ProjectPapersResponse:
    papers = call_application(
        lambda: service.add_papers(
            user_id=user_id,
            project_id=project_id,
            paper_ids=request.paper_ids,
        )
    )
    return ProjectPapersResponse(
        papers=[PaperResponse.model_validate(item) for item in papers]
    )


@router.delete(
    "/{project_id}/papers/{paper_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def remove_project_paper(
    project_id: UUID,
    paper_id: UUID,
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    service: Annotated[ProjectService, Depends(get_project_service)],
) -> Response:
    call_application(
        lambda: service.remove_paper(
            user_id=user_id,
            project_id=project_id,
            paper_id=paper_id,
        )
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_project(
    project_id: UUID,
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    service: Annotated[ProjectService, Depends(get_project_service)],
) -> Response:
    call_application(
        lambda: service.delete(user_id=user_id, project_id=project_id)
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)

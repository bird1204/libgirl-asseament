    require 'rubygems/package'
    require 'zlib'

class UploadsController < ApplicationController
  def new
    @project = Project.new
  end

  def create
    # @project = Project.create(project_params)
    @project = Project.last
    # url = "http://#{request.host}:#{request.port}#{rails_blob_path(@project.source)}"
    # system("curl -s #{url} | tar zxvf -C /Users/wei-yichiu/workspace/libgirl-hello")
    #tempfile = @project.source.download_blob_to("/Users/wei-yichiu/workspace/#{@project.source.filename}")

    binary = @project.source.download
    File.open("/Users/wei-yichiu/workspace/#{@project.source.filename}", "wb") do |file|
      file.write(binary)
    end

    system("tar -zxvf /Users/wei-yichiu/workspace/#{@project.source.filename}")
    system("cd /Users/wei-yichiu/workspace/libgirl-hello && rails s -p 5000")

    redirect_to 'http://lvh.me:5000'
  end

  private

  def project_params
    params.require(:project).permit(:source)
  end
end

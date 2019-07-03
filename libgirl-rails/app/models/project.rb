class Project < ApplicationRecord
  include ActiveStorage::Downloading
  has_one_attached :source


  def download
    
  end
end

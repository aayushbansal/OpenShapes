function [img_pat_loc] = get_img_pat_loc(crop_height, crop_width, pat)

        img_pat_loc = false(crop_height*crop_width, crop_height*crop_width);

        iter = 1;
        for j = 1:crop_width
                for i = 1:crop_height

                        img_pat = zeros(crop_height, crop_width);
                        img_pat(i,j) = true;

                        st_pos_i = min(max(i - pat, 1), crop_height);
                        st_pos_j = min(max(j - pat, 1), crop_width);
                        end_pos_i = min(max(st_pos_i+2*pat,1), crop_height);
                        end_pos_j = min(max(st_pos_j+2*pat,1), crop_width);
				
                        img_pat(st_pos_i:end_pos_i, st_pos_j:end_pos_j) = true;
                        img_pat_loc(iter,:) = img_pat(:);
                        iter = iter+1;
                end
        end
end
